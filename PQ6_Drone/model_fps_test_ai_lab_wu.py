"""
Перед тестом:
sudo nvpmodel -m 0
sudo jetson_clocks

Проверить:
tegrastats

GPU должен работать на 1100 MHz без throttling

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT YOLO Benchmark 
—Автор: AI Lab/Hennadii
-версия для Jetson Orin NX
Описание:
- jetson_utils.videoSource (CSI → GPU)
- cudaToNumpy + torch GPU preprocessing (FP16)
- Одна копия .cpu().numpy() → pinned host → TensorRT
- execute_async_v2 + stream
"""

import time
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

try:
    import jetson_utils
except ImportError:
    print("\n[ERROR] jetson_utils не установлен!")
    print("sudo pip3 install jetson-utils")
    sys.exit(1)


class JetsonTRTBenchmarkJetsonUtils:
    def __init__(self, engine_path: str, imgsz: int = 960, force_fp16: bool = True):
        self.imgsz = imgsz
        self.force_fp16 = force_fp16
        self.np_dtype = np.float16 if force_fp16 else np.float32

        print("\n" + "═" * 80)
        print(" TensorRT + jetson_utils (стабильный GPU pipeline) ".center(80))
        print("═" * 80)

        # --------------------- TensorRT Engine ---------------------
        engine_path = Path(engine_path)
        if not engine_path.exists():
            print(f"[ERROR] Engine не найден: {engine_path}")
            sys.exit(1)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            print(f"engine_path = {f}")
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_idx = self.engine.get_binding_index("images") or 0
        self.output_idx = self.engine.get_binding_index("output0") or 1

        self.input_shape = self.engine.get_binding_shape(self.input_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_idx)

        vol_in = trt.volume(self.input_shape)
        vol_out = trt.volume(self.output_shape)

        # Pinned host memory
        self.input_host = cuda.pagelocked_empty(vol_in, dtype=self.np_dtype)
        self.output_host = cuda.pagelocked_empty(vol_out, dtype=np.float32)

        # Device memory
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)

        self.stream = cuda.Stream()

        print(f"[OK] TensorRT engine загружен")
        print(f"   Input : {self.input_shape} ({self.np_dtype.__name__})")
        print(f"   Output: {self.output_shape}\n")

    def warmup(self, iterations=50):
        print("─" * 80)
        print(" ПРОГРЕВ GPU ".center(80))
        print("─" * 80)

        dummy_np = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        dummy_cuda = jetson_utils.cudaFromNumpy(dummy_np)

        for i in range(iterations):
            _ = self.infer(dummy_cuda)
            if (i + 1) % 10 == 0:
                print(f"[warmup] {i+1:3d}/{iterations}")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[warmup] завершён\n")

    def preprocess(self, cuda_img):
        """cuda_img → numpy (zero-copy) → torch GPU → FP16 → numpy для TRT"""
        # Zero-copy mapped numpy
        np_img = jetson_utils.cudaToNumpy(cuda_img)          # HWC uint8

        # Torch на GPU
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).to(device="cuda", dtype=torch.float32)
        tensor = tensor / 255.0

        # Resize (если нужно)
        if tensor.shape[1:] != (self.imgsz, self.imgsz):
            tensor = F.interpolate(tensor.unsqueeze(0), size=(self.imgsz, self.imgsz),
                                   mode='bilinear', align_corners=False).squeeze(0)

        # FP16
        if self.force_fp16:
            tensor = tensor.half()

        return tensor.contiguous().cpu().numpy()              # ← одна копия на CPU

    def infer(self, cuda_img):
        img_input = self.preprocess(cuda_img)                # CHW fp16/fp32 numpy

        np.copyto(self.input_host, img_input.ravel())
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)

        bindings = [int(self.input_device), int(self.output_device)]
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        return self.output_host.copy()

    def benchmark_stream(self, sensor_id=0, iterations=500, fps=30):
        print("\n" + "═" * 80)
        print(" Бенчмарк CSI → jetson_utils → TensorRT ".center(80))
        print("═" * 80)

        camera = jetson_utils.videoSource(f"csi://{sensor_id}",
                                          ["--input-width=3840",
                                           "--input-height=2160",
                                           f"--input-fps={fps}"])

        read_times, infer_times, total_times = [], [], []

        try:
            for i in range(iterations):
                t0 = time.perf_counter()
                img_cuda = camera.Capture()
                if img_cuda is None:
                    continue
                t1 = time.perf_counter()

                _ = self.infer(img_cuda)
                t2 = time.perf_counter()

                read_times.append(t1 - t0)
                infer_times.append(t2 - t1)
                total_times.append(t2 - t0)

                if (i + 1) % 50 == 0:
                    print(f"[stream] {i+1:4d}/{iterations}", end="\r")
        except KeyboardInterrupt:
            print("\nОстановлено пользователем")
        finally:
            camera.Close()

        if not total_times:
            print("Кадры не получены")
            return

        read_ms = np.mean(read_times) * 1000
        infer_ms = np.mean(infer_times) * 1000
        total_ms = np.mean(total_times) * 1000

        print("\n" + "─" * 80)
        print(" РЕЗУЛЬТАТЫ ".center(80))
        print("─" * 80)
        print(f" Захват кадра      : {read_ms:6.2f} мс")
        print(f" Inference         : {infer_ms:6.2f} мс")
        print(f" Полный цикл       : {total_ms:6.2f} мс")
        print(f" Pipeline FPS      : {1000 / total_ms:6.2f}")
        print(f" Чистый FPS модели : {1000 / infer_ms:6.2f}")
        print("═" * 80 + "\n")


if __name__ == "__main__":
    model_dir = Path("/home/user/yolo_tests/PQ6_Drone")
    engine_files = list(model_dir.glob("*.engine"))
    if not engine_files:
        print("Не найдено *.engine файлов")
        sys.exit(1)

    print("Рекомендуется перед запуском:")
    print("   sudo nvpmodel -m 0")
    print("   sudo jetson_clocks\n")

    for engine in engine_files:
        print(f"\nТестируем: {engine.name}")
        print("─" * 80)
        bench = JetsonTRTBenchmarkJetsonUtils(str(engine), imgsz=960, force_fp16=True)
        bench.warmup()
        bench.benchmark_stream(sensor_id=0, iterations=500, fps=30)
