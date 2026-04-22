import cv2
import time

class FPSCounter:
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
    
    def start(self):
        self.start_time = time.time()
        self.frame_count = 0
    
    def update(self):
        self.frame_count += 1
        current_time = time.time()
        
        # Обновляем FPS каждую секунду
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.start_time = current_time
            self.frame_count = 0
            self.prev_time = current_time
        
        return self.fps

def display_fps_on_video(camera_id=0):
    """
    Показывает видео с текущим FPS
    """
    cap = cv2.VideoCapture(camera_id)
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    fps_counter = FPSCounter()
    fps_counter.start()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обновляем счетчик FPS
        current_fps = fps_counter.update()
        
        # Отображаем FPS на кадре
        cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Показываем кадр
        cv2.imshow('Camera with FPS', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Запуск
display_fps_on_video()