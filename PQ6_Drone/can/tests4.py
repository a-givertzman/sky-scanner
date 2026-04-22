import cv2
import time
from ultralytics import YOLO

def display_fps_on_video(camera_id=1):
    """
    Показывает видео с текущим FPS и YOLO12n
    """
    # Загружаем модель YOLO12n
    model = YOLO('/home/user/yolo_tests/PQ6_Drone/yolo12n_SP_5072_opz_960.pt')
    model.to("cuda")
    
    cap = cv2.VideoCapture(camera_id)
    
    while True:
        # Замеряем время перед чтением кадра
        read_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Время чтения кадра
        read_time = time.time() - read_start_time
        
        # Замеряем время перед обработкой YOLO
        process_start_time = time.time()
        
        # Обработка кадра с помощью YOLO
        results = model(frame, conf=0.3, verbose=False)
        
        # Время обработки YOLO
        process_time = time.time() - process_start_time
        
        # Выводим только два параметра
        print(f"Считывание: {read_time*1000:.1f} мс, Обработка: {process_time*1000:.1f} мс")
        
        # Показываем кадр с результатами
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO12n', annotated_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Запуск
if __name__ == "__main__":
    display_fps_on_video(camera_id=1)