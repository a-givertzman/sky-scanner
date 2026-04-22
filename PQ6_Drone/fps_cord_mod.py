import threading, queue, time
import cv2, numpy as np, torch
from ultralytics import YOLO
from pymodbus.server.sync import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
import struct
import os
import sys

MODEL_PATH = "/home/user/yolo_tests/PQ6_Drone/yolo26n_drone_960_half.engine"
CAM_ID = 0
INPUT_W, INPUT_H = 1280, 1280
MODE = "tensor_fp16"
QSIZE = 2

# Modbus сервер
class ModbusServer:
    def __init__(self, host="localhost", port=5020):
        self.host = host
        self.port = port
        self.context = None
        self.server_thread = None
        self.store = None
        self.registers = [0] * 10000  # Локальное хранилище
        
    def start(self):
        """Запуск Modbus TCP сервера"""
        try:
            # Создаем хранилище данных
            self.store = ModbusSlaveContext(
                zeroMode=True,
                hr=ModbusSequentialDataBlock(0, self.registers)
            )
            
            # Создаем контекст сервера
            self.context = ModbusServerContext(slaves=self.store, single=True)
            
            # Запускаем сервер в отдельном потоке
            self.server_thread = threading.Thread(
                target=StartTcpServer,
                kwargs={
                    'context': self.context,
                    'address': (self.host, self.port),
                    'defer_start': False  # Важно для синхронного режима
                },
                daemon=True
            )
            self.server_thread.start()
            print(f"Modbus server started on {self.host}:{self.port}")
            time.sleep(1)  # Даем время серверу запуститься
            
        except Exception as e:
            print(f"Modbus server start error: {e}")
            
    def update_objects(self, results):
        """Обновление данных объектов в Modbus регистрах"""
        try:
            if not self.store:
                return
                
            num_objects = len(results[0].boxes) if results and len(results) > 0 else 0
            
            # Регистр 0: количество объектов
            self.registers[0] = num_objects
            
            # Очищаем старые данные
            for i in range(1, 221):
                self.registers[i] = 0
            
            if num_objects > 0:
                for i, box in enumerate(results[0].boxes):
                    if i >= 10:
                        break
                        
                    # Получаем данные
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # ID класса
                    self.registers[1 + i] = class_id
                    
                    # Целые координаты
                    base_int = 11 + i * 10
                    self.registers[base_int] = int(x1)
                    self.registers[base_int + 1] = int(y1)
                    self.registers[base_int + 2] = int(x2)
                    self.registers[base_int + 3] = int(y2)
                    
                    # Float координаты
                    base_float = 111 + i * 8
                    x1_regs = self.float_to_registers(x1)
                    y1_regs = self.float_to_registers(y1)
                    x2_regs = self.float_to_registers(x2)
                    y2_regs = self.float_to_registers(y2)
                    
                    if base_float + 7 < len(self.registers):
                        self.registers[base_float] = x1_regs[0]
                        self.registers[base_float + 1] = x1_regs[1]
                        self.registers[base_float + 2] = y1_regs[0]
                        self.registers[base_float + 3] = y1_regs[1]
                        self.registers[base_float + 4] = x2_regs[0]
                        self.registers[base_float + 5] = x2_regs[1]
                        self.registers[base_float + 6] = y2_regs[0]
                        self.registers[base_float + 7] = y2_regs[1]
                    
                    # Уверенность
                    conf_regs = self.float_to_registers(conf)
                    base_conf = 211 + i * 2
                    if base_conf + 1 < len(self.registers):
                        self.registers[base_conf] = conf_regs[0]
                        self.registers[base_conf + 1] = conf_regs[1]
            
            # Обновляем данные в Modbus хранилище
            # Важно: обновляем через setValues, а не напрямую
            self.store.setValues(0x00, 0, self.registers[0:1000])
            
        except Exception as e:
            print(f"Error updating Modbus: {e}")
    
    @staticmethod
    def float_to_registers(value):
        """Конвертация float в два 16-битных регистра"""
        try:
            packed = struct.pack('>f', float(value))
            return [int.from_bytes(packed[0:2], 'big'), 
                    int.from_bytes(packed[2:4], 'big')]
        except:
            return [0, 0]

# Modbus клиент для тестирования
class ModbusTestClient:
    def __init__(self, host="localhost", port=5020):
        self.host = host
        self.port = port
        self.client = None
        
    def connect(self):
        try:
            from pymodbus.client.sync import ModbusTcpClient
            self.client = ModbusTcpClient(self.host, self.port)
            connected = self.client.connect()
            if connected:
                print("Modbus client connected")
            return connected
        except Exception as e:
            print(f"Failed to connect Modbus client: {e}")
            return False
    
    def read_objects(self):
        """Чтение данных объектов из Modbus"""
        if not self.client:
            return
            
        try:
            # Читаем количество объектов
            result = self.client.read_holding_registers(0, 1, unit=0x00)
            if result.isError():
                return
                
            num_objects = result.registers[0]
            
            if num_objects > 0:
                print(f"\n=== Modbus данные: {num_objects} объектов ===")
                
                # Читаем ID классов
                result = self.client.read_holding_registers(1, 10, unit=0x00)
                class_ids = result.registers if not result.isError() else []
                    
                for i in range(min(num_objects, 10)):
                    # Читаем целые координаты
                    base_int = 11 + i * 10
                    result = self.client.read_holding_registers(base_int, 4, unit=0x00)
                    
                    if not result.isError() and len(result.registers) >= 4:
                        x1, y1, x2, y2 = result.registers[:4]
                        
                        # Читаем float координаты
                        base_float = 111 + i * 8
                        result_float = self.client.read_holding_registers(base_float, 8, unit=0x00)
                        
                        # Читаем уверенность
                        base_conf = 211 + i * 2
                        result_conf = self.client.read_holding_registers(base_conf, 2, unit=0x00)
                        
                        print(f"\nОбъект {i+1}:")
                        if i < len(class_ids):
                            print(f"  Class ID: {class_ids[i]}")
                        print(f"  Целые координаты: ({x1}, {y1}, {x2}, {y2})")
                        
                        if not result_float.isError() and len(result_float.registers) >= 8:
                            x1_f = self.registers_to_float(result_float.registers[0:2])
                            y1_f = self.registers_to_float(result_float.registers[2:4])
                            x2_f = self.registers_to_float(result_float.registers[4:6])
                            y2_f = self.registers_to_float(result_float.registers[6:8])
                            print(f"  Float координаты: x1={x1_f:.2f}, y1={y1_f:.2f}, x2={x2_f:.2f}, y2={y2_f:.2f}")
                        
                        if not result_conf.isError() and len(result_conf.registers) >= 2:
                            conf = self.registers_to_float(result_conf.registers[0:2])
                            print(f"  Уверенность: {conf:.3f}")
            else:
                # Если объектов нет, просто показываем что живы
                pass
                
        except Exception as e:
            print(f"Error reading Modbus: {e}")
    
    @staticmethod
    def registers_to_float(regs):
        """Конвертация двух регистров в float"""
        try:
            if len(regs) >= 2:
                packed = regs[0].to_bytes(2, 'big') + regs[1].to_bytes(2, 'big')
                return struct.unpack('>f', packed)[0]
        except:
            pass
        return 0.0
    
    def close(self):
        if self.client:
            self.client.close()

def main():
    # Инициализация модели
    model = YOLO(MODEL_PATH)

    # Инициализация камеры
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("Camera initialized")
    
    # Очередь для кадров
    frame_queue = queue.Queue(maxsize=2)
    stopped = False
    
    # Запуск Modbus сервера
    modbus = ModbusServer(host="localhost", port=5020)
    modbus.start()
    
    # Тестовый клиент
    test_client = ModbusTestClient(host="localhost", port=5020)
    if test_client.connect():
        print("Test client connected")
    
    # Функция захвата кадров
    def capture_frames():
        nonlocal stopped
        while not stopped:
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(frame)
                    except:
                        pass
            else:
                time.sleep(0.001)
    
    # Запуск потока
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    time.sleep(0.5)
    
    # Прогрев
    try:
        frame = frame_queue.get(timeout=2.0)
        if frame is not None:
            _ = model(frame, verbose=False)
    except:
        pass
    
    frame_count = 0
    fps_time = time.time()
    fps_counter = 0
    
    try:
        while True:
            t0 = time.time()
            
            # Получение кадра
            try:
                frame = frame_queue.get(timeout=0.1)
                if frame is None:
                    continue
            except queue.Empty:
                continue
            
            # Инференс
            try:
                results = model(frame, verbose=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Inference error: {e}")
                continue
            
            # Отправка в Modbus
            if modbus:
                modbus.update_objects(results)
            
            # Тестовое чтение
            frame_count += 1
            if frame_count % 30 == 0 and test_client:
                test_client.read_objects()
            
            # FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                print(f"FPS: {fps_counter}")
                fps_counter = 0
                fps_time = time.time()
            
            # Отображение
            annotated = results[0].plot()
            cv2.putText(annotated, f"Objects: {len(results[0].boxes)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            cv2.imshow("Stream", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        stopped = True
        cap.release()
        cv2.destroyAllWindows()
        if test_client:
            test_client.close()
        print("Cleanup done")

if __name__ == "__main__":
    main()