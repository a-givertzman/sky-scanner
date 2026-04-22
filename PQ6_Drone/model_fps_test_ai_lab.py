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
Production benchmark для TensorRT YOLO на Jetson Orin NX
Оптимизирован для стабильного измерения latency и FPS
Автор: AI Lab/Hennadii
Описание:
- jetson_utils используется для дополнительного мониторинга Jetson (если доступен): температура, мощность, частоты.
- GStreamer pipeline дополнительно оптимизирован: добавлен format=NV12 для NVMM, capsfilter для стабильности
- Добавлена проверка на наличие тестового изображения в main.
- Улучшена обработка ошибок: больше sys.exit только при фатальных ошибках, return при не-фатальных.
- Добавлен мониторинг в warmup и benchmarks если jetson_utils available.
- Защита от деления на 0 расширена.
- Graceful shutdown в stream: ловит exceptions.
- Убраны потенциальные проблемы с памятью: добавлен gc.collect() после empty_cache().
- Исправлен tegrastats: используем --interval 1000 --count 1 для однократного вывода.
- Удален cudaDeviceSynchronize из log_resources.
- Добавлен imgsz в вызовы model().
- Добавлена проверка размера кадра в benchmark_stream.
- Удален torch.backends.cudnn.benchmark = True (бесполезно для TensorRT engine).
- Добавлен флаг ENABLE_SYSTEM_LOGS = False для отключения логов ресурсов (избегаем задержек от tegrastats).
- В GStreamer: videoconvert n-threads=2 для снижения CPU нагрузки на Orin NX.
"""

import cv2
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import gc
import subprocess

# Флаг для включения системных логов (tegrastats и т.д.) - по умолчанию False, чтобы избежать задержек
ENABLE_SYSTEM_LOGS = True

# ----------------------------------------------------------
# Безопасный импорт jetson_utils
# ----------------------------------------------------------
try:
    import jetson_utils
    JETSON_UTILS_AVAILABLE = True
except ImportError:
    JETSON_UTILS_AVAILABLE = False
    print("[WARNING] jetson_utils не доступен. Установите для полного мониторинга: pip install jetson-utils")

# ==========================================================
# Оптимизированный GStreamer pipeline (с NVMM и capsfilter для стабильности)
# ==========================================================
def gstreamer_pipeline(sensor_id=0, width=960, height=960, fps=30):
    """
    - NVMM память до ресайза.
    - capsfilter для предотвращения неявных конвертаций.
    - max-buffers=1 для минимальной latency.
    - sync=false drop=true для асинхронного высокоскоростного чтения.
    - videoconvert n-threads=2 для снижения CPU нагрузки.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=3840, height=2160, framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12 ! "
        f"capsfilter caps=video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert n-threads=2 ! "  # Уменьшено до 2 потоков для снижения CPU нагрузки
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true sync=false max-buffers=1"
    )

def gstreamer_pipeline_usb(width=960, height=960, fps=30, device="/dev/video0"):
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
        f"videoconvert n-threads=2 ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true sync=false max-buffers=1"
    )

# ==========================================================
# Benchmark класс
# ==========================================================
class JetsonTRTBenchmark:
    def __init__(self, engine_path: str, imgsz: int = 960):
        """
        Инициализация:
        - Проверка CUDA.
        - Проверка существования engine.
        - Загрузка модели с try-except.
        - Очистка кэша GPU + gc для освобождения памяти.
        """
        self.imgsz = imgsz
        print("\n==============================================")
        print("ИНИЦИАЛИЗАЦИЯ МОДЕЛИ")
        print("==============================================")
        if not torch.cuda.is_available():
            print("[ERROR] CUDA не доступен. Проверьте драйверы и Jetson setup.")
            sys.exit(1)
        engine_path = Path(engine_path)
        if not engine_path.exists():
            print(f"[ERROR] Engine файл не найден: {engine_path}")
            sys.exit(1)
        print(f"[INFO] Загрузка TensorRT engine: {engine_path}")
        try:
            self.model = YOLO(str(engine_path))
        except Exception as e:
            print(f"[ERROR] Не удалось загрузить модель: {e}")
            sys.exit(1)
        torch.cuda.empty_cache()
        gc.collect()
        print("[OK] Модель успешно инициализирована на GPU\n")

    # ------------------------------------------------------
    # Логирование GPU/системных ресурсов
    # ------------------------------------------------------
    def log_resources(self):
        """
        Логирование:
        - GPU память через torch.cuda.
        - Если ENABLE_SYSTEM_LOGS и jetson_utils: tegrastats с минимальной задержкой.
        Это помогает диагностировать bottlenecks (overheating, throttling).
        - tegrastats: однократный вызов с --interval 1000 --count 1 (но только если флаг включен).
        """
        if not ENABLE_SYSTEM_LOGS:
            return  # Пропускаем, чтобы избежать задержек
        print(f"[LOG] Ресурсы системы:")
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            print(f"  GPU память:")
            print(f"    Выделено PyTorch: {allocated / 1024**2:.2f} MB")
            print(f"    Свободно: {free_mem / 1024**2:.2f} MB")
            print(f"    Всего: {total_mem / 1024**2:.2f} MB")
        except Exception as e:
            print(f"  [WARNING] GPU память: {e}")
        
        if JETSON_UTILS_AVAILABLE:
            try:
                # Для stats используем tegrastats напрямую
                tegra_output = subprocess.check_output(["tegrastats", "--interval", "1000", "--count", "1"], text=True).strip()
                print(f"  Jetson stats (tegrastats): {tegra_output}")
            except Exception as e:
                print(f"  [WARNING] tegrastats: {e}")
        print("")

    # ------------------------------------------------------
    # Warmup
    # ------------------------------------------------------
    def warmup(self, iterations=50):
        """
        Прогрев:
        - Dummy input для стабилизации TensorRT.
        - Логи ресурсов до/после.
        - Прогресс каждые 10 итераций.
        - Добавлен imgsz в model call.
        """
        print("==============================================")
        print("ПРОГРЕВ GPU")
        print("==============================================")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.log_resources()
        with torch.no_grad():
            for i in range(iterations):
                _ = self.model(dummy, imgsz=self.imgsz, verbose=False)
                if (i + 1) % 10 == 0:
                    print(f"[LOG] Прогрев: {i+1}/{iterations}")
        torch.cuda.synchronize()
        self.log_resources()
        print(f"[OK] Прогрев завершён ({iterations} итераций)\n")

    # ------------------------------------------------------
    # Чистый inference benchmark
    # ------------------------------------------------------
    def benchmark_inference(self, image_path: str, iterations=300):
        """
        Бенчмарк инференса:
        - Проверка изображения.
        - Ресайз.
        - Измерение с sync.
        - Статистика с защитой от 0.
        - Логи ресурсов.
        - Добавлен imgsz в model call.
        """
        print("==============================================")
        print("БЕНЧМАРК ЧИСТОГО ИНФЕРЕНСА")
        print("==============================================")
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[ERROR] Изображение не найдено: {image_path}")
            return
        img = cv2.imread(str(image_path))
        if img is None:
            print("[ERROR] Не удалось загрузить изображение.")
            return
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        latencies = []
        self.log_resources()
        with torch.no_grad():
            for i in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.model(img, imgsz=self.imgsz, verbose=False)
                torch.cuda.synchronize()
                end = time.perf_counter()
                latency = end - start
                latencies.append(latency)
                if (i + 1) % 50 == 0:
                    print(f"[LOG] Инференс: {i+1}/{iterations}")
        if len(latencies) == 0:
            print("[ERROR] Нет измерений.")
            return
        latencies = np.array(latencies)
        self.log_resources()
        mean_latency = latencies.mean()
        print("\n---------- РЕЗУЛЬТАТЫ ----------")
        print(f"Средняя задержка: {mean_latency*1000:.2f} мс")
        print(f"Минимальная задержка: {latencies.min()*1000:.2f} мс")
        print(f"Максимальная задержка: {latencies.max()*1000:.2f} мс")
        print(f"Std отклонение: {latencies.std()*1000:.2f} мс")
        if mean_latency > 0:
            print(f"Средний FPS модели: {1/mean_latency:.2f}")
        else:
            print("Средний FPS модели: N/A")
        print("----------------------------------\n")

    # ------------------------------------------------------
    # Streaming benchmark
    # ------------------------------------------------------
    def benchmark_stream(self, sensor_id=0, iterations=500):
        """
        Бенчмарк стрима:
        - GStreamer пайплайн.
        - Try-except для KeyboardInterrupt и других ошибок.
        - Finally: release cap.
        - Статистика с защитой.
        - Логи ресурсов.
        - Добавлен imgsz в model call.
        - Добавлена проверка и resize кадра если размер не совпадает.
        """
        print("==============================================")
        print("БЕНЧМАРК ПОТОКОВОГО ВИДЕО (USB V4L2)")
        print("==============================================")
        
        # Заменяем  на V4L2
        cap = cv2.VideoCapture(0)  # 0 = /dev/video0, или 1 = /dev/video1 и т.д.
        
        # Настраиваем параметры (разрешение, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)    
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("[ERROR] Не удалось открыть USB-камеру (/dev/video0 или другой индекс).")
            print("Проверьте: v4l2-ctl --list-devices")
            return
        read_times = []
        infer_times = []
        total_times = []
        self.log_resources()
        print("[INFO] Начало измерений...\n")
        try:
            with torch.no_grad():
                for i in range(iterations):
                    r_start = time.perf_counter()
                    ret, frame = cap.read()
                    r_end = time.perf_counter()
                    if not ret:
                        print(f"[WARNING] Кадр не получен ({i+1})")
                        continue
                    if frame.shape[0] != self.imgsz or frame.shape[1] != self.imgsz:
                        frame = cv2.resize(frame, (self.imgsz, self.imgsz))
                    torch.cuda.synchronize()
                    i_start = time.perf_counter()
                    _ = self.model(frame, imgsz=self.imgsz, verbose=False)
                    torch.cuda.synchronize()
                    i_end = time.perf_counter()
                    read_time = r_end - r_start
                    infer_time = i_end - i_start
                    total_time = read_time + infer_time
                    read_times.append(read_time)
                    infer_times.append(infer_time)
                    total_times.append(total_time)
                    if (i + 1) % 50 == 0:
                        print(f"[LOG] Обработано кадров: {i+1}/{iterations}")
        except KeyboardInterrupt:
            print("\n[INFO] Остановка пользователем.")
        except Exception as e:
            print(f"\n[ERROR] Ошибка в стриме: {e}")
        finally:
            cap.release()
            self.log_resources()
        if len(total_times) == 0:
            print("[ERROR] Нет успешных кадров.")
            return
        read_times = np.array(read_times)
        infer_times = np.array(infer_times)
        total_times = np.array(total_times)
        mean_read = read_times.mean()
        mean_infer = infer_times.mean()
        mean_total = total_times.mean()
        print("\n---------- РЕЗУЛЬТАТЫ ----------")
        print(f"Среднее время чтения: {mean_read*1000:.2f} мс")
        print(f"Среднее время инференса: {mean_infer*1000:.2f} мс")
        print(f"Средний pipeline latency: {mean_total*1000:.2f} мс")
        if mean_total > 0:
            print(f"Pipeline FPS: {1/mean_total:.2f}")
        else:
            print("Pipeline FPS: N/A")
        if mean_infer > 0:
            print(f"Чистый FPS модели: {1/mean_infer:.2f}")
        else:
            print("Чистый FPS модели: N/A")
        print("----------------------------------\n")

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    """
    Main:
    - Проверка директории и изображения.
    - Рекомендации для power mode.
    - Цикл по engines.
    - Для включения системных логов: установите ENABLE_SYSTEM_LOGS = True
    """
    MODEL_DIR = Path("/home/user/yolo_tests/PQ6_Drone")
    TEST_IMAGE = Path("/home/user/yolo_tests/PQ6_Drone/test_img.jpg")
    if not MODEL_DIR.exists() or not MODEL_DIR.is_dir():
        print(f"[ERROR] Директория не найдена или не директория: {MODEL_DIR}")
        sys.exit(1)
    if not TEST_IMAGE.exists():
        print(f"[ERROR] Тестовое изображение не найдено: {TEST_IMAGE}")
        sys.exit(1)
    engines = list(MODEL_DIR.glob("*.engine"))
    if not engines:
        print(f"[ERROR] Не найдено .engine файлов в {MODEL_DIR}")
        sys.exit(1)
    print("[INFO] Рекомендуется запуск в MAXN режиме:")
    print("       sudo nvpmodel -m 0")
    print("       sudo jetson_clocks\n")
    for engine in engines:
        print("\n===================================================")
        print(f"ТЕСТИРОВАНИЕ МОДЕЛИ: {engine.name}")
        print("===================================================")
        benchmark = JetsonTRTBenchmark(engine, imgsz=960)
        benchmark.warmup()
        benchmark.benchmark_inference(TEST_IMAGE)
        benchmark.benchmark_stream(sensor_id=1)
