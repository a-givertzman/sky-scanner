import threading
import queue
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pymodbus.server.sync import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.client.sync import ModbusTcpClient

# Конфигурация
MODEL_PATH = "/home/user/yolo_tests/PQ6_Drone/distillation_YOLO26s_YOLO26n.engine"
# MODEL_PATH = "/home/user/yolo_tests/PQ6_Drone/yolo26n_drone_640_half.engine"

CAM_ID = 0
INPUT_W, INPUT_H = 1280, 1280
QSIZE = 2

# Modbus регистры:
# 0: center_x (центр по X)
# 1: center_y (центр по Y)

def run_modbus_server():
    """Запуск Modbus TCP сервера в отдельном потоке"""
    store = ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0] * 100),
        co=ModbusSequentialDataBlock(0, [0] * 100),
        ir=ModbusSequentialDataBlock(0, [0] * 100),
        hr=ModbusSequentialDataBlock(0, [0] * 100)
    )
    
    context = ModbusServerContext(slaves=store, single=True)
    
    identity = ModbusDeviceIdentification()
    identity.VendorName = 'YOLO Detection Server'
    
    print("Modbus сервер запущен на порту 5020")
    StartTcpServer(context=context, identity=identity, address=("0.0.0.0", 5020))

def main():
    # Запускаем Modbus сервер
    modbus_thread = threading.Thread(target=run_modbus_server, daemon=True)
    modbus_thread.start()
    time.sleep(1)
    
    # Подключаемся к серверу
    client = ModbusTcpClient('127.0.0.1', port=5020)
    if not client.connect():
        print("Ошибка подключения к Modbus серверу")
        return
    
    print("Подключено к Modbus серверу")
    
    # Загрузка модели
    model = YOLO(MODEL_PATH)
    try:
        model.to("cuda")
        print("Модель на GPU")
    except:
        print("Модель на CPU")
    
    # Камера
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    
    # Очередь для фреймов
    frame_queue = queue.Queue(maxsize=QSIZE)
    stopped = False
    
    def capture_frames():
        nonlocal stopped
        while not stopped:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
    
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    time.sleep(0.3)
    
    # Прогрев модели
    frame = None
    try:
        frame = frame_queue.get(timeout=1.0)
    except queue.Empty:
        print("No camera frame.")
        stopped = True
        cap.release()
        return
    
    inp_frame = cv2.resize(frame, (INPUT_W, INPUT_H))
    for _ in range(10):
        _ = model(inp_frame, verbose=False)
    
    print("Начинаем детекцию. Нажмите 'q' для выхода.\n")
    
    try:
        while True:
            t0 = time.time()
            
            try:
                frame = frame_queue.get(timeout=None)
            except queue.Empty:
                continue
            
            t1 = time.time()
            
            # Детекция
            results = model(frame, verbose=False, half=True, end2end=False)
            
            t2 = time.time()
            
            # Обработка результатов
            if len(results[0].boxes) > 0:
                # Берем первый объект
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Отправляем в Modbus
                client.write_registers(0, [center_x, center_y], unit=1)
                
                # Читаем обратно из Modbus для проверки
                result = client.read_holding_registers(0, 2, unit=1)
                if not result.isError():
                    read_x = result.registers[0]
                    read_y = result.registers[1]
                    print(f"Отправил: ({center_x}, {center_y}) | Прочитал: ({read_x}, {read_y})")
            else:
                # Если нет объектов, ничего не отправляем
                print("Объектов не обнаружено")
            
            t3 = time.time()
            
            # Расчет FPS
            fps = 1.0 / max((t3 - t0), 1e-6)
            
            # Визуализация с FPS
            annotated = results[0].plot()
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"Capture: {(t1 - t0) * 1000:.1f} ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Inference: {(t2 - t1) * 1000:.1f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Modbus: {(t3 - t2) * 1000:.1f} ms", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Total: {(t3 - t0) * 1000:.1f} ms", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Detection", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        stopped = True
        client.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nЗавершено")

if __name__ == "__main__":
    main()