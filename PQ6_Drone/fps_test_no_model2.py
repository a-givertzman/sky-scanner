import cv2

# Открываем камеру
cap = cv2.VideoCapture(0)  # или другой индекс, если камер несколько

# 1. Попробуйте установить формат FOURCC на MJPEG перед установкой разрешения
# Это критически важный шаг для многих камер[citation:9][citation:10]
mjpeg_fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
success = cap.set(cv2.CAP_PROP_FOURCC, mjpeg_fourcc)

if not success:
    print("Внимание: Не удалось установить формат MJPEG. FPS может быть низким.")

# 2. Теперь устанавливаем высокое разрешение и частоту кадров
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
# Попытка установить FPS (не всегда поддерживается, но стоит попробовать)
cap.set(cv2.CAP_PROP_FPS, 60)

# 3. Проверяем, что получилось
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Запрошено: 3840x2160 @ 60 FPS")
print(f"Получено: {int(actual_width)}x{int(actual_height)} @ {actual_fps:.2f} FPS")

# 4. Простой тест производительности
import time
frame_count = 0
start_time = time.time()

# Измеряем FPS в течение 5 секунд
while (time.time() - start_time) < 5:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # Можно отображать изображение, но это замедлит работу
    # cv2.imshow('Test', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

end_time = time.time()
measured_fps = frame_count / (end_time - start_time)
print(f"Измеренный FPS: {measured_fps:.2f}")

cap.release()
cv2.destroyAllWindows()