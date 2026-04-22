import torch
import time
import numpy as np
from ultralytics import YOLO

def accurate_benchmark(model_path='yolo11n.pt', warmup=20, runs=100):
    """Точный бенчмарк с правильной предобработкой данных."""
    
    # Загрузка модели
    print(f"Загрузка модели {model_path}...")
    model = YOLO(model_path)
    model.to('cuda')
    model.eval()
    model.fuse()  # Оптимизация модели для инференса
    
    # Создаем реалистичные данные (нормализованные 0-1, как ожидает модель)
    # Размер batch=1, 3 канала, 640x640
    dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
    dummy_input = dummy_input.float() / 255.0  # КРИТИЧНО: нормализация!
    
    # Альтернатива: реальное изображение для точности
    # from PIL import Image
    # import torchvision.transforms as T
    # img = Image.open('test.jpg')
    # transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    # dummy_input = transform(img).unsqueeze(0).to('cuda')
    
    print(f"Прогрев ({warmup} итераций)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input, verbose=False)
    
    # Замер с высокоточным таймером
    print(f"Бенчмарк ({runs} итераций)...")
    timings = []
    
    with torch.no_grad():
        for i in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(dummy_input, verbose=False)
            end.record()
            
            # Ждем завершения всех ядер CUDA
            torch.cuda.synchronize()
            
            elapsed = start.elapsed_time(end)  # время в миллисекундах
            timings.append(elapsed)
            
            if (i + 1) % 25 == 0:
                print(f"  Итерация {i+1:3d}: {elapsed:.2f} мс")
    
    # Статистика
    timings_np = np.array(timings)
    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТЫ ДЛЯ {model_path}")
    print(f"{'='*50}")
    print(f"Среднее время инференса: {timings_np.mean():.2f} ± {timings_np.std():.2f} мс")
    print(f"Медиана:                 {np.median(timings_np):.2f} мс")
    print(f"Минимум:                 {timings_np.min():.2f} мс")
    print(f"Максимум:                {timings_np.max():.2f} мс")
    print(f"95-й перцентиль:         {np.percentile(timings_np, 95):.2f} мс")
    print(f"Теоретический FPS:       {1000 / timings_np.mean():.0f}")
    
    # Проверка использования памяти
    if hasattr(torch.cuda, 'memory_stats'):
        mem = torch.cuda.memory_stats()
        peak_mem = mem.get('allocated_bytes.all.peak', 0) / 1024**2
        print(f"Пиковое использование памяти: {peak_mem:.1f} MB")

if __name__ == "__main__":
    # Тестируйте последовательно
    print("="*60)
    print("ТЕСТ 1: Официальная модель YOLOv11n")
    print("="*60)
    accurate_benchmark(model_path='yolo11n.pt')
    
    print("\n" + "="*60)
    print("ТЕСТ 2: Ваша кастомная модель")
    print("="*60)
    # accurate_benchmark(model_path='/home/user/yolo_tests/PQ6_Drone/yolo12n_SP_5072_opz_960.pt')