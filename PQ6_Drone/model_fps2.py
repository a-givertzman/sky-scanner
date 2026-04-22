# stream_infer.py
import threading, queue, time
import cv2, numpy as np, torch
from ultralytics import YOLO

MODEL_PATH = "/home/user/yolo_tests/PQ6_Drone/yolo26n_drone_960_half.engine"
CAM_ID = 0
INPUT_W, INPUT_H = 1280, 1280
MODE = "tensor_fp16"   # or "tensor_fp16"
QSIZE = 2

class CameraThread(threading.Thread):
    def __init__(self, cam_id=0, w=3840, h=2160, qsize=2):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.q = queue.Queue(maxsize=qsize)
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self.q.get_nowait()
                    self.q.put_nowait(frame)
                except queue.Empty:
                    pass

    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.cap.release()

def preprocess(frame, w, h):
    # Только изменение размера до (w, h). Ни нормализации, ни перестановки каналов.
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame

def main():
    model = YOLO(MODEL_PATH)
    try:
        model.to("cuda")
    except Exception:
        pass

    dtype = torch.float16 if MODE == "tensor_fp16" else torch.float32

    #w=3840, h=2160

    #w=1920, h=1080

    cam_thread = CameraThread(CAM_ID, w=1920, h=1080, qsize=QSIZE)
    cam_thread.start()
    time.sleep(0.3)

    # Warmup with one frame
    frame = cam_thread.read()
    if frame is None:
        print("No camera frame.")
        cam_thread.stop()
        return

    inp_frame = preprocess(frame, INPUT_W, INPUT_H)
    for _ in range(10):
        _ = model(inp_frame, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    fps_smooth = 0.0
    alpha = 0.9

    try:
        while True:
            t0 = time.time()
            frame = cam_thread.read(timeout=None)
            # ret, frame = cam_thread.cap.read()
            t1 = time.time()
            if frame is None:
                continue
            # inp_frame = preprocess(frame, INPUT_W, INPUT_H)
            t2 = time.time()
            # Передаём numpy BGR кадр напрямую в модель
            results = model(frame, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.time()
            # annotated = frame.copy()
            
            annotated = results[0].plot()
            fps = 1.0 / max((t3 - t0), 1e-6)
            fps_smooth = alpha * fps_smooth + (1 - alpha) * fps

            # Рисуем FPS перед показом кадра
            if annotated is None:
                annotated = frame.copy()
            # print(f"FPS: {fps:.1f}")
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"R_time: {((t1-t0)*1000):.1f}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"C_time: {((t2-t1)*1000):.1f}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"M_time: {((t3-t2)*1000):.1f}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"R+C+M: {((t3-t0)*1000):.1f} or {((t1-t0)*1000 + (t2-t1)*1000 + (t3-t2)*1000):.1f}", (10,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            if ((t3-t0)*1000 == (t1-t0)*1000 + (t2-t1)*1000 + (t3-t2)*1000):
                cv2.putText(annotated, f"OK", (10,210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            cv2.imshow("Stream", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam_thread.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
