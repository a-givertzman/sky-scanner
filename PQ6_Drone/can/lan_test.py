import cv2
import time

# ЕДИНСТВЕННОЕ, ЧТО НУЖНО ПОМЕНЯТЬ — ЭТО ЭТОТ URL
# Используем RTSP, порт 554 (он у тебя открыт)
url = "rtsp://admin:admin@192.168.100.99:554/stream1"
# Если не работает, попробуй:
# url = "rtsp://admin:admin@192.168.100.99:554/live/main"
# url = "rtsp://admin:admin@192.168.100.99:554/stream0"

print(f"Подключаюсь к {url}...")
cap = cv2.VideoCapture(url)

# Проверка подключения
if not cap.isOpened():
    print("❌ Не удалось подключиться!")
    print("Проверь:")
    print("  - IP камеры (ping 192.168.100.99)")
    print("  - Логин/пароль (admin/admin)")
    exit()

print("✅ Подключено! Жду кадры...")
# time.sleep(1)  # Даём время на установку потока

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Потерян поток")
        break
    
    # Считаем FPS
    frame_count += 1
    if time.time() - start_time >= 1:
        print(f"FPS: {frame_count}")
        frame_count = 0
        start_time = time.time()
    
    # Показываем видео
    cv2.imshow('Camera', frame)
    
    
    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()