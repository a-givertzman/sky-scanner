import cv2
import time
import torch
from ultralytics import YOLO

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

def display_fps_on_video(camera_id=1):
    """
    Показывает видео с текущим FPS и YOLO12n
    """
    # Загружаем модель YOLO12n
    print("Загрузка модели YOLO12n...")
    # try:
    model = YOLO('/home/user/yolo_tests/PQ6_Drone/yolo12n_SP_5072_opz_960.pt')  # или 'yolo12n.pt' для вашей обученной модели
    # except:
    #     # Если нет локальной модели, загружаем предобученную
    #     # model = YOLO('yolo12n.pt')
    # print("Модель загружена")
    
    cap = cv2.VideoCapture(camera_id)
    fps_counter = FPSCounter()
    fps_counter.start()
    
    # Переменные для измерения времени
    last_frame_time = time.time()
    frame_number = 0
    
    while True:
        # Замеряем время перед чтением кадра
        read_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Время чтения кадра
        read_end_time = time.time()
        frame_read_time = read_end_time - read_start_time
        
        # Замеряем время перед обработкой YOLO
        process_start_time = time.time()
        
        # Обработка кадра с помощью YOLO
        results = model(frame, verbose=False)
        
        # Время обработки YOLO
        process_end_time = time.time()
        process_time = process_end_time - process_start_time
        
        # Отображение результатов YOLO на кадре
        annotated_frame = results[0].plot()
        
        # Получаем информацию о детекциях
        detections = results[0].boxes
        num_detections = len(detections) if detections is not None else 0
        
        # Интервал между кадрами
        current_time = time.time()
        frame_interval = current_time - last_frame_time
        last_frame_time = current_time
        
        # Обновляем счетчик FPS
        current_fps = fps_counter.update()
        
        # Выводим в консоль информацию
        frame_number += 1
        print(f"Кадр {frame_number}: "
              f"Чтение = {frame_read_time*1000:.1f} мс, "
              f"Обработка = {process_time*1000:.1f} мс, "
              f"Всего = {(frame_read_time + process_time)*1000:.1f} мс, "
              f"Интервал = {frame_interval*1000:.1f} мс, "
              f"FPS = {current_fps:.1f}, "
              f"Объектов = {num_detections}")
        
        # Отображаем информацию на кадре
        y_offset = 30
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Read: {frame_read_time*1000:.1f} ms", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(annotated_frame, f"Process: {process_time*1000:.1f} ms", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 0, 255), 2)
        
        y_offset += 25
        cv2.putText(annotated_frame, f"Total: {(frame_read_time + process_time)*1000:.1f} ms", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(annotated_frame, f"Objects: {num_detections}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 200, 255), 2)
        
        # Показываем кадр
        cv2.imshow('YOLO12n + FPS', annotated_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Завершено")

# Расширенная версия с усреднением статистики
def display_fps_on_video_advanced(camera_id=1):
    """
    Расширенная версия с усреднением статистики
    """
    # Загружаем модель
    print("Загрузка YOLO12n...")
    model = YOLO('yolo12n.pt')
    print("Модель готова")
    
    cap = cv2.VideoCapture(camera_id)
    fps_counter = FPSCounter()
    fps_counter.start()
    
    # Статистика
    read_times = []
    process_times = []
    total_times = []
    frame_number = 0
    
    # Для вывода каждые N кадров
    stats_interval = 10
    
    while True:
        # === ЧТЕНИЕ КАДРА ===
        read_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        read_time = time.time() - read_start
        
        # === ОБРАБОТКА YOLO ===
        process_start = time.time()
        results = model(frame, verbose=False)
        process_time = time.time() - process_start
        
        # === РИСОВАНИЕ РЕЗУЛЬТАТОВ ===
        annotated_frame = results[0].plot()
        detections = results[0].boxes
        num_detections = len(detections) if detections is not None else 0
        
        # === СТАТИСТИКА ===
        total_time = read_time + process_time
        read_times.append(read_time)
        process_times.append(process_time)
        total_times.append(total_time)
        
        frame_number += 1
        current_fps = fps_counter.update()
        
        # Вывод статистики каждые N кадров
        if frame_number % stats_interval == 0:
            avg_read = sum(read_times[-stats_interval:]) / min(stats_interval, len(read_times))
            avg_process = sum(process_times[-stats_interval:]) / min(stats_interval, len(process_times))
            avg_total = sum(total_times[-stats_interval:]) / min(stats_interval, len(total_times))
            
            print(f"[{frame_number}] "
                  f"FPS: {current_fps:.1f} | "
                  f"Чтение: {avg_read*1000:.1f}мс | "
                  f"YOLO: {avg_process*1000:.1f}мс | "
                  f"Всего: {avg_total*1000:.1f}мс | "
                  f"Объекты: {num_detections}")
        
        # === ОТОБРАЖЕНИЕ НА КАДРЕ ===
        # Создаем панель с информацией
        info_height = 150
        info_panel = annotated_frame.copy()
        
        # Фон для текста
        cv2.rectangle(info_panel, (0, 0), (300, info_height), (0, 0, 0), -1)
        cv2.addWeighted(info_panel, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
        # Текст с метриками
        metrics = [
            f"Frame: {frame_number}",
            f"FPS: {current_fps:.1f}",
            f"Read: {read_time*1000:.1f} ms",
            f"YOLO: {process_time*1000:.1f} ms",
            f"Total: {total_time*1000:.1f} ms",
            f"Objects: {num_detections}"
        ]
        
        for i, metric in enumerate(metrics):
            color = (0, 255, 0) if i == 1 else (255, 255, 255)  # FPS зеленым
            cv2.putText(annotated_frame, metric,
                       (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)
        
        # === ПОКАЗ КАДРА ===
        cv2.imshow('YOLO12n Real-time Detection', annotated_frame)
        
        # Выход
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Финальная статистика
    if read_times:
        print("\n" + "="*60)
        print("ФИНАЛЬНАЯ СТАТИСТИКА:")
        print(f"Всего кадров: {frame_number}")
        print(f"Среднее время чтения: {sum(read_times)/len(read_times)*1000:.1f} мс")
        print(f"Среднее время обработки: {sum(process_times)/len(process_times)*1000:.1f} мс")
        print(f"Среднее общее время: {sum(total_times)/len(total_times)*1000:.1f} мс")
        print(f"Средний FPS: {1000/(sum(total_times)/len(total_times)*1000):.1f}")
        print("="*60)
    
    cap.release()
    cv2.destroyAllWindows()

# Запуск
if __name__ == "__main__":
    # Базовая версия
    display_fps_on_video(camera_id=1)
    
    # Или расширенная версия
    # display_fps_on_video_advanced(camera_id=1)