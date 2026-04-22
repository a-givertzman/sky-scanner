import cv2
import time
from ultralytics import YOLO

def display_fps_on_video(camera_index=0):
    # model = YOLO('/home/user/yolo_tests/PQ6_Drone/yolo12n_SP_5072_opz_960.pt')
    model = YOLO('/home/user/yolo_tests/PQ6_Drone/yolo26n_drone_960.pt')

    # model = YOLO('yolo11n.pt')
    # try:
    #     # model.to("cuda")
    #     print("Model moved to CUDA.")
    # except Exception:
    #     print("CUDA unavailable or move failed; using default device.")
    cap = cv2.VideoCapture(camera_index)  # Для Linux/Mac

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'H265')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # Попробуйте установить разрешение 4K
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Установлено разрешение: {actual_width}x{actual_height}")
    while True:
        r_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # frame = frame[0:640,0:640]
        r_time = time.time() - r_start
        m_start = time.time()
        results = model(frame, verbose=False)
        m_time = time.time() - m_start
        fps = 1/(r_time+m_time)
        height, width = frame.shape[:2]
        print(f"Размер: {width}x{height} | Считывание: {r_time*1000:.1f} мс, Обработка: {m_time*1000:.1f} мс, FPS: {fps:.1f}")
        annotated_frame = results[0].plot()
        print(f"Кадр: {width}x{height}, Каналов: {frame.shape[2] if len(frame.shape) > 2 else 1}")

        if width > 1920:
            display_frame = cv2.resize(annotated_frame, (1920, 1080))
        else:
            display_frame = annotated_frame
        
        cv2.imshow('Camera', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_fps_on_video(camera_index=0)
