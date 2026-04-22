import cv2
import time
from ultralytics import YOLO

def display_fps_on_video(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    start = time.time()
    frames = 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    while True:
        r_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        r_time = time.time() - r_start
        frames += 1
        curr = time.time()
        if curr - start >= 1.0:
            f_fps = frames / (time.time() - start)
            start = time.time()
            frames = 0
            print(f"F_FPS: {f_fps:.1f}")

        fps = 1/r_time
        height, width = frame.shape[:2]
        print(f"Размер: {width}x{height} | Считывание: {r_time*1000:.1f} мс, FPS: {fps:.1f}")
        cv2.imshow('YOLO12n', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_fps_on_video(camera_id=0)

    # if current_time - self.prev_time >= 1.0:
    #         self.fps = self.frame_count / (current_time - self.start_time)
    #         self.start_time = current_time
    #         self.frame_count = 0
    #         self.prev_time = current_time