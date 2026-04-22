import threading, queue, time
import cv2, numpy as np, torch
from ultralytics import YOLO

MODEL_PATH = "/home/user/yolo_tests/PQ6_Drone/distillation_YOLO26s_YOLO26n.engine"
CAM_ID = 0
INPUT_W, INPUT_H = 960, 960
MODE = "tensor_fp16"
QSIZE = 2



def main():
    model = YOLO(MODEL_PATH)
    try:
        model.to("cuda")
    except Exception:
        pass

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    frame_queue = queue.Queue(maxsize=QSIZE)
    stopped = False

    def capture_frames():
        nonlocal stopped
        while not stopped:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
    

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    time.sleep(0.3)

    frame = None
    try:
        frame = frame_queue.get(timeout=1.0)
    except queue.Empty:
        print("No camera frame.")
        stopped = True
        cap.release()
        return
    
    inp_frame = cv2.resize(frame, (INPUT_W, INPUT_H))
    for _ in range(10):
        _ = model(inp_frame, verbose=False)
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()

    try:
        while True:
            t0 = time.time()
            try:
                frame = frame_queue.get(timeout=None)
            except queue.Empty:
                continue
            t1 = time.time()

            results = model(frame, verbose=True, half = True, end2end = False)
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            t2 = time.time()
            
            annotated = results[0].plot()
            fps = 1.0 / max((t2 - t0), 1e-6)
            
            # Вывод координат объектов в консоль
            if len(results[0].boxes) > 0:
                print(f"\n--- Обнаружено {len(results[0].boxes)} объектов ---")
                for i, box in enumerate(results[0].boxes):
                    # Координаты bounding box (xyxy формат)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = results[0].names[class_id]
                    
                    print(f"Объект {i+1}: {class_name}")
                    print(f"  Координаты: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                    print(f"  Центр: ({((x1+x2)/2):.1f}, {((y1+y2)/2):.1f})")
                    print(f"  Ширина: {(x2-x1):.1f}, Высота: {(y2-y1):.1f}")
                    print(f"  Уверенность: {confidence:.3f}")
            
            # Отображение FPS и времени на кадре
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"Capture: {((t1-t0)*1000):.1f} ms", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated, f"Inference: {((t2-t1)*1000):.1f} ms", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated, f"Total: {((t2-t0)*1000):.1f} ms", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow("Stream", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        stopped = True
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()