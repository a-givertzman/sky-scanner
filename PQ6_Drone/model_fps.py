from pathlib import Path
import cv2
import time
import numpy as np
from ultralytics import YOLO

def display_fps_on_video(model_name, test_type, size = 960, image_path = None, camera_id=0):
    model = YOLO(model_name)
    print(f"Model device is: {model.device}")
    try:
        model.to("cuda")
        print("Model moved to CUDA.")
    except Exception:
        print("CUDA unavailable or move failed; using default device.")
    

    match test_type:
        case 0:
            cap = cv2.VideoCapture(camera_id)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            fps_counter = 0
            fps10 = 0
            model10 = 0
            read10 = 0

            for _ in range(50):
                ret, frame = cap.read()
                _ = model(frame, verbose=False)

            for i in range(100):
                r_start = time.time()
                ret, frame = cap.read()
                r_time = time.time() - r_start
                if not ret:
                    break
                # frame = frame[0:size,0:size]
                m_start = time.time()
                results = model(frame, verbose=False)
                m_time = time.time() - m_start
                fps = 1/(r_time+m_time)
                height, width = frame.shape[:2]
                if fps_counter != 10:
                    fps_counter = fps_counter + 1
                    fps10 = fps10 + fps
                    model10 = model10 + m_time
                    read10 = read10 + r_time
                else:
                    fps_counter = 0
                    fps10 = fps10 / 10
                    model10 = model10 / 10
                    read10 = read10 / 10
                    print(f"Считывание: {read10*1000:.1f} мс, Обработка: {model10*1000:.1f} мс, FPS: {fps10:.1f}")



                # print(f"Размер: {width}x{height} | Считывание: {r_time*1000:.1f} мс, Обработка: {m_time*1000:.1f} мс, FPS: {fps:.1f}")
                # annotated_frame = results[0].plot()
                # cv2.imshow('Result', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        case 1:
            r_start = time.time()
            img = cv2.imread(str(image_path))
            r_time = time.time() - r_start
            print(r_time)
            if img.shape[0] > size or img.shape[1] > size:
                img = img[0:size, 0:size]
            
            print(f"Img size: {img.shape[1]}x{img.shape[0]}")
            
            # model heat up
            for _ in range(50):
                _ = model(img, verbose=False)
            
            total_inference_times = []
            total_fps = []
            
            print("\nЗапуск 100 итераций...")
            for i in range(100):
                start_time = time.perf_counter()
                results = model(img, verbose=False)
                inference_time = time.perf_counter() - start_time
                
                total_inference_times.append(inference_time)
                fps = 1.0 / inference_time if inference_time > 0 else 0
                total_fps.append(fps)
                
                if (i + 1) % 10 == 0:
                    print(f"Iter {i+1}/100: {inference_time*1000:.1f} мс, FPS: {fps:.1f}")
            
            # Статистика
            print("\n" + "="*50)
            print("Results")
            print("="*50)
            
            # Время обработки
            avg_inference_time = np.mean(total_inference_times)
            min_inference_time = np.min(total_inference_times)
            max_inference_time = np.max(total_inference_times)
            std_inference_time = np.std(total_inference_times)
            
            print(f"\nModel times:")
            print(f"  Mean: {avg_inference_time*1000:.1f} мс")
            print(f"  Min: {min_inference_time*1000:.1f} мс")
            print(f"  Max: {max_inference_time*1000:.1f} мс")
            print(f"  Std: {std_inference_time*1000:.1f} мс")
            
            # FPS
            avg_fps = np.mean(total_fps)
            min_fps = np.min(total_fps)
            max_fps = np.max(total_fps)
            
            print(f"\nFPS:")
            print(f"  Mean: {avg_fps:.1f}")
            print(f"  Min: {min_fps:.1f}")
            print(f"  Max: {max_fps:.1f}")
            

            # annotated_img = results[0].plot()
            # cv2.imshow('Результат детекции', annotated_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            

if __name__ == "__main__":
    model_dir = Path("/home/user/yolo_tests/PQ6_Drone")
    engine_files = list(model_dir.glob("*.engine"))

    for model_path in engine_files:
        print(f"\nTesting: {model_path.name}")
        
        # model_name = "/home/user/yolo_tests/PQ6_Drone/yolo12n_SP_5072_opz_960.engine"
        model_name = str(model_path)
        display_fps_on_video(model_name, test_type=0, size = 960)
        display_fps_on_video(model_name, test_type=1, size = 960, image_path="/home/user/yolo_tests/PQ6_Drone/test_img.jpg")
