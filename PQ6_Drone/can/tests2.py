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
    fps_counter = FPSCounter()
    fps_counter.start()
    
    # Переменные для измерения времени чтения кадра
    last_frame_time = time.time()
    frame_number = 0
    
    while True:
        # Замеряем время перед чтением кадра
        read_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Замеряем время после чтения кадра
        read_end_time = time.time()
        frame_read_time = read_end_time - read_start_time
        
        # Интервал между кадрами
        current_time = time.time()
        frame_interval = current_time - last_frame_time
        last_frame_time = current_time
        
        # Обновляем счетчик FPS
        current_fps = fps_counter.update()
        
        # Выводим в консоль информацию о времени чтения кадра
        frame_number += 1
        print(f"Кадр {frame_number}: "
              f"Время чтения = {frame_read_time*1000:.2f} мс, "
              f"Интервал = {frame_interval*1000:.2f} мс, "
              f"FPS = {current_fps:.1f}")
        
        # Отображаем FPS на кадре
        cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Отображаем время чтения кадра на видео
        cv2.putText(frame, f"Read time: {frame_read_time*1000:.1f} ms", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        # Показываем кадр
        cv2.imshow('Camera with FPS', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

display_fps_on_video()