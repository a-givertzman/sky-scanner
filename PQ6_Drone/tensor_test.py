import cv2
import time
from ultralytics import YOLO

def display_fps_on_video(camera_id=0):
    # model = YOLO('/home/user/yolo_tests/PQ6_Drone/yolo12n_SP_5072_opz_960.pt')
    model = YOLO('/home/user/yolo_tests/dist/models_to_test/yolo26n_1280.pt')
    # # model = YOLO('yolo11n.pt')
    model.export(format="engine", half = True)
    trt_model = YOLO("/home/user/yolo_tests/PQ6_Drone/yolo26n_1280.engine", task = "detect")
    # trt_model = YOLO("yolo11n.engine")
    
    # try:
    #     trt_model.to("cuda")
    #     print("Model moved to CUDA.")
    # except Exception:
    #     print("CUDA unavailable or move failed; using default device.")
    cap = cv2.VideoCapture(camera_id)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # Попробуйте установить разрешение 4K
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    while True:
        r_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[0:960,0:960]
        r_time = time.time() - r_start
        m_start = time.time()
        results = trt_model(frame, verbose=False)
        m_time = time.time() - m_start
        fps = 1/(r_time+m_time)
        height, width = frame.shape[:2]
        print(f"Размер: {width}x{height} | Считывание: {r_time*1000:.1f} мс, Обработка: {m_time*1000:.1f} мс, FPS: {fps:.1f}")
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO12n', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_fps_on_video(camera_id=0)