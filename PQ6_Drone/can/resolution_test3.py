import cv2
import time

def open_camera_4k_mjpg(device=0):
    """
    Открывает камеру с принудительным использованием MJPG кодека для 4K
    """
    # Открываем камеру с V4L2 бэкендом (для Linux)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return None
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Устанавливаем разрешение 4K
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    print(f"Запрошено: 3840x2160")
    print(f"Установлено: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps}")
    print(f"FourCC кодек: {chr(actual_fourcc & 0xFF)}{chr((actual_fourcc >> 8) & 0xFF)}{chr((actual_fourcc >> 16) & 0xFF)}{chr((actual_fourcc >> 24) & 0xFF)}")
    
    # Даем камере время на инициализацию
    time.sleep(1.0)
    
    # Пробуем получить кадр
    for i in range(5):  # Несколько попыток
        ret, frame = cap.read()
        if ret:
            print(f"Успех! Размер кадра: {frame.shape}")
            return cap
        else:
            print(f"Попытка {i+1}: не удалось получить кадр")
            time.sleep(0.5)
    
    print("Не удалось получить кадр после нескольких попыток")
    cap.release()
    return None

# Использование
cap = open_camera_4k_mjpg(0)
if cap:
    # Сохраняем тестовый кадр
    ret, frame = cap.read()
    if ret:
        # cv2.imwrite('4k_test_mjpg.jpg', frame)
        print(f"Сохранено изображение размером: {frame.shape}")
    
    cap.release()