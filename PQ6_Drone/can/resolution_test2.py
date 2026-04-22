import cv2

# Попробуйте разные индексы камеры
camera_index = 0  # 0, 1, 2 и т.д.

#cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Для Windows
cap = cv2.VideoCapture(camera_index)  # Для Linux/Mac

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# Попробуйте установить разрешение 4K
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)



# Проверьте, какое разрешение установилось
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Установлено разрешение: {actual_width}x{actual_height}")

# # Если 4K не устанавливается, попробуйте другие разрешения
# if actual_width < 3840:
#     print("Попробуем Full HD...")
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
#     actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     print(f"Установлено: {actual_width}x{actual_height}")

# Чтение кадров
print("Нажмите 'q' для выхода")
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Ошибка чтения кадра")
        break
    
    # Показываем информацию о кадре
    height, width = frame.shape[:2]
    print(f"Кадр: {width}x{height}, Каналов: {frame.shape[2] if len(frame.shape) > 2 else 1}")
    
    # Масштабируем для отображения (если 4K)
    if width > 1920:
        display_frame = cv2.resize(frame, (1920, 1080))
    else:
        display_frame = frame
    
    cv2.imshow('Camera', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()