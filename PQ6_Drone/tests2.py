import threading
import time
from pymodbus.server.sync import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.client.sync import ModbusTcpClient

def run_server():
    store = ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0] * 100),
        co=ModbusSequentialDataBlock(0, [0] * 100),
        ir=ModbusSequentialDataBlock(0, [0] * 100),
        hr=ModbusSequentialDataBlock(0, [0] * 100)
    )
    
    store.setValues(3, 0, [100, 200, 300, 400, 500])
    
    context = ModbusServerContext(slaves=store, single=True)
    
    identity = ModbusDeviceIdentification()
    identity.VendorName = 'Python Modbus Server'
    
    print("Запуск Modbus TCP сервера на порту 5020...")
    StartTcpServer(context=context, identity=identity, address=("127.0.0.1", 5020))

# Запускаем сервер в отдельном потоке
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Даем серверу время запуститься
time.sleep(1)

print("123123")

# Подключаемся к серверу
client = ModbusTcpClient('127.0.0.1', port=5020)

if client.connect():
    print("Подключено")
    
    # Читаем регистры
    result = client.read_holding_registers(0, 10)
    if not result.isError():
        print(f"Значения регистров: {result.registers}")
    
    # Пишем в регистр
    client.write_register(0, 9999)
    print("Записали 9999 в регистр 0")
    
    # Читаем еще раз
    result = client.read_holding_registers(0, 10)
    print(f"Новые значения: {result.registers}")
    
    client.close()
else:
    print("Не удалось подключиться")

# Держим программу активной
input("Нажми Enter для выхода...\n")