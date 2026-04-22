import torch
import time

tensor = torch.randn(5000, 5000).cuda()

# Без synchronize()
start = time.time()
for _ in range(10):
    tensor = tensor @ tensor
end = time.time()
print(f"Без синхронизации: {end-start:.4f} сек")  # Очень маленькое число!

# С synchronize()
start = time.time()
for _ in range(10):
    tensor = tensor @ tensor
torch.cuda.synchronize()
end = time.time()
print(f"С синхронизацией: {end-start:.4f} сек")  # Реальное время