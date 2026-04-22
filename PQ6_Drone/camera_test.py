import cv2
import time

url = "rtsp://admin:admin@192.168.100.99:554/stream1"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Не удалось подключиться")
    exit()

print("Подключено! Ожидание кадров...")

# Просто читаем кадры, не обращая внимания на cap.get()
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    if ret and frame is not None:
        # Получаем реальное разрешение из кадра
        h, w = frame.shape[:2]
        
        # Считаем FPS
        frame_count += 1
        if time.time() - start_time >= 1:
            print(f"FPS: {frame_count} | Разрешение: {w}x{h}")
            frame_count = 0
            start_time = time.time()
        
        # Показываем
        cv2.imshow('Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Нет кадра")
        time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()