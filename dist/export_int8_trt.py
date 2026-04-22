from ultralytics import YOLO

# === PATHS ===
MODEL_PATH = r"/home/user/yolo_tests/dist/distillation_YOLO26s_YOLO26n.pt" # Путь к обученой модели YOLO для экспорта в TensorRT INT8
DATA_YAML = r"/home/user/yolo_tests/dist/calibration_int8_dataset/data_calibration.yaml" # Путь к YAML-файлу калибровочного датасета

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === EXPORT TO TENSORRT INT8 ===
model.export(
    format="engine",        # TensorRT
    imgsz=960,              # размер обучения
    int8=True,              # включаем INT8
    data=DATA_YAML,         # calibration dataset
    device=0,               # GPU
    simplify=False,
    end2end = False,
    batch = 1,
)

print("\nINT8 TensorRT export completed.")