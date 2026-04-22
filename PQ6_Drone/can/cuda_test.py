import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"Архитектура: {torch.__file__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"CUDA версия: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

import onnx
print('ONNX version:', onnx.__version__)

import tensorrt as trt
print('TensorRT version:', trt.__version__)