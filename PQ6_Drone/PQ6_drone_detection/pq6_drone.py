# PQ6 Drone Detector (PyQt6) with YOLO, Kalman + OpticalFlow, sound alerts
import sys
import os
import time
import threading
import queue
import ctypes
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from PyQt6 import QtCore, QtGui, QtWidgets
import pygame  # pip install pygame

# ==================================================================
# Конфигурация и классы
# ==================================================================
# Словарь с именами классов, которые распознаёт модель YOLO
CLASS_NAMES = {
    0: "DRONE",      # Дрон
    1: "AIRPLANE",   # Самолёт
    2: "HELICOPTER", # Вертолёт
    3: "BIRDS"       # Птицы
}

# ==================================================================
# Утилиты
# ==================================================================
def get_short_path(path: str) -> str:
    """Возвращает короткий путь в формате 8.3 для Windows (помогает некоторым сборкам OpenCV)."""
    try:
        buf = ctypes.create_unicode_buffer(4096)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 4096)
        if res != 0:
            return buf.value
    except Exception:
        pass
    return str(path)

# ==================================================================
# Надёжное открытие видео
# ==================================================================
def open_video_safely(path: str, log_cb=None, timeout_sec=2.0):
    """
    Пробует открыть видеофайл с помощью разных бэкендов OpenCV.
    log_cb(msg) — callback для вывода логов.
    Возвращает cv2.VideoCapture или None.
    """
    def log(m):
        if log_cb:
            try:
                log_cb(m)
            except Exception:
                pass

    if path is None:
        log("[ERROR] open_video_safely: path is None")
        return None

    p = Path(path)
    if not p.exists():
        log(f"[ERROR] File not found: {path}")
        return None

    try_path = get_short_path(path)

    backends = []
    if hasattr(cv2, "CAP_FFMPEG"):
        backends.append(("CAP_FFMPEG", cv2.CAP_FFMPEG))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("CAP_MSMF", cv2.CAP_MSMF))
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("CAP_DSHOW", cv2.CAP_DSHOW))
    # CAP_ANY как резервный вариант
    backends.append(("CAP_ANY", cv2.CAP_ANY if hasattr(cv2, "CAP_ANY") else 0))

    for name, api in backends:
        log(f"[INFO] Trying backend: {name}")
        try:
            cap = cv2.VideoCapture(try_path, api)
            t0 = time.time()
            while time.time() - t0 < timeout_sec:
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        log(f"[INFO] Opened with {name} (first frame OK)")
                    else:
                        log(f"[WARN] Opened with {name} but failed to read first frame")
                    # Переоткрываем чисто для вызывающего кода
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap2 = cv2.VideoCapture(try_path, api)
                    if cap2.isOpened():
                        return cap2
                    else:
                        try:
                            cap2.release()
                        except Exception:
                            pass
                    break
                time.sleep(0.05)
            try:
                cap.release()
            except Exception:
                pass
            log(f"[WARN] Backend {name} failed to open")
        except Exception as e:
            log(f"[ERROR] Backend {name} exception: {e}")

    # Резервный вариант без указания бэкенда
    try:
        log("[INFO] Trying fallback cv2.VideoCapture(path) without backend flag")
        cap = cv2.VideoCapture(try_path)
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if cap.isOpened():
                ret, _ = cap.read()
                try:
                    cap.release()
                except Exception:
                    pass
                if ret:
                    return cv2.VideoCapture(try_path)
                break
            time.sleep(0.05)
        try:
            cap.release()
        except Exception:
            pass
    except Exception as e:
        log(f"[ERROR] Fallback exception: {e}")

    log("[ERROR] All backends failed. Likely codec/format not supported by this OpenCV build.")
    log("Recommendation: convert to H.264 Baseline MP4 or install ffmpeg-enabled OpenCV / system ffmpeg.")
    return None

# ==================================================================
# Воспроизведение звука (pygame)
# ==================================================================
class SoundPlayer:
    """Класс для воспроизведения звукового оповещения при обнаружении объекта."""
    def __init__(self):
        # Инициализируем микшер лениво, чтобы не падать при отсутствии аудиоустройства
        try:
            pygame.mixer.init()
            self.available = True
        except Exception:
            self.available = False
        self.sound = None
        self.cooldown = 0.5  # секунд между воспроизведениями
        self.last_play = 0.0
        self.volume = 1.0

    def load(self, path):
        """Загружает звуковой файл."""
        if not self.available:
            return False, "pygame.mixer not initialized"
        try:
            self.sound = pygame.mixer.Sound(path)
            self.sound.set_volume(self.volume)
            return True, None
        except Exception as e:
            return False, str(e)

    def set_volume(self, v: float):
        """Устанавливает громкость."""
        self.volume = float(v)
        try:
            if self.sound:
                self.sound.set_volume(self.volume)
        except Exception:
            pass

    def play(self):
        """Воспроизводит звук с защитой от слишком частого вызова."""
        if not self.available or self.sound is None:
            return
        now = time.time()
        if now - self.last_play > self.cooldown:
            try:
                self.sound.play()
                self.last_play = now
            except Exception:
                pass

# ==================================================================
# Прокси для логирования
# ==================================================================
class SimpleLoggerProxy:
    """Простой обёрток для безопасного вызова callback логирования."""
    def __init__(self, cb):
        self.cb = cb

    def __call__(self, msg):
        try:
            self.cb(msg)
        except Exception:
            pass

# ==================================================================
# Рабочий поток: инференс YOLO + трекинг (Kalman + Optical Flow) + звук
# ==================================================================
class WorkerThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)  # RGB ndarray для GUI
    status = QtCore.pyqtSignal(str)              # Сообщения в лог

    def __init__(self, model_path, source_spec, conf, track_selected, sound_player: SoundPlayer, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.source_spec = source_spec.copy()
        self.conf = float(conf)
        self.track_selected = track_selected.copy()
        self.sound_player = sound_player
        self._stop = threading.Event()

    def stop(self):
        """Сигнал остановки потока."""
        self._stop.set()

    def log(self, msg):
        """Отправка сообщения в лог GUI."""
        try:
            self.status.emit(msg)
        except Exception:
            pass

    def run(self):
        # --------------------- Загрузка модели YOLO ---------------------
        try:
            self.log("[INFO] Loading YOLO model...")
            model = YOLO(self.model_path)
            # Пытаемся перенести модель на GPU
            try:
                model.to("cuda")
                self.log("[INFO] Model moved to CUDA.")
            except Exception:
                self.log("[WARN] CUDA unavailable or move failed; using default device.")
        except Exception as e:
            self.log(f"[ERROR] Model load failed: {e}")
            return

        # --------------------- Открытие источника видео ---------------------
        cap = None
        if self.source_spec["type"] == "file":
            path = self.source_spec["value"]
            self.log(f"[INFO] Opening file: {path}")
            cap = open_video_safely(path, log_cb=SimpleLoggerProxy(self.log))
            if cap is None or not cap.isOpened():
                self.log("[ERROR] Capture not opened for file. Worker exiting.")
                return
        else:  # camera
            idx = int(self.source_spec["value"])
            self.log(f"[INFO] Opening camera index {idx}")
            for api in [cv2.CAP_ANY,
                        getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
                        getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)]:
                try:
                    cap = cv2.VideoCapture(idx, api)
                    time.sleep(0.05)
                    if cap.isOpened():
                        break
                except Exception:
                    cap = None
            if cap is None or not cap.isOpened():
                self.log("[ERROR] Cannot open camera.")
                return
            self.log("[INFO] Camera opened.")

        # --------------------- Параметры потока ---------------------
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        except Exception:
            w, h, fps = 640, 480, 30.0
        self.log(f"[INFO] Stream: {w}x{h} @ {fps:.1f}fps")

        # Состояние трекинга
        prev_gray = None
        prev_point = None
        kalman = None
        kalman_init = False
        lost = 0
        MAX_LOST = 20
        frame_count = 0
        t_last = time.time()

        # --------------------- Основной цикл обработки кадров ---------------------
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                self.log("[INFO] No frame (end of stream or read error).")
                break

            # Подготовка серого изображения для Optical Flow
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = None

            # --------------------- Инференс YOLO ---------------------
            try:
                results = model(frame, conf=self.conf, verbose=False)
                res = results[0]
            except Exception as e:
                self.log(f"[ERROR] Inference error: {e}")
                break

            dets = res.boxes  # Объект Boxes из ultralytics
            detected = False
            measure = None
            detected_class = None

            # --------------------- Обработка детекций ---------------------
            if dets is not None and len(dets) > 0:
                try:
                    xyxy_arr = dets.xyxy.cpu().numpy()
                    cls_arr = dets.cls.cpu().numpy().astype(np.int32)
                except Exception:
                    # Резервный способ получения боксов
                    xyxy_arr = []
                    cls_arr = []
                    for b in dets:
                        try:
                            xy = b.xyxy.cpu().numpy()[0]
                            ci = int(b.cls.cpu().numpy()[0])
                            xyxy_arr.append(xy)
                            cls_arr.append(ci)
                        except Exception:
                            pass
                    if len(xyxy_arr) > 0:
                        xyxy_arr = np.array(xyxy_arr)
                        cls_arr = np.array(cls_arr, dtype=np.int32)

                if len(xyxy_arr) > 0:
                    for box, cid in zip(xyxy_arr, cls_arr):
                        cid_int = int(cid)
                        if self.track_selected.get(cid_int, False):
                            x1, y1, x2, y2 = box[:4]
                            cx = float((x1 + x2) / 2.0)
                            cy = float((y1 + y2) / 2.0)
                            measure = np.array([[cx], [cy]], dtype=np.float32)
                            detected = True
                            detected_class = cid_int
                            # Воспроизведение звука
                            try:
                                self.sound_player.play()
                            except Exception:
                                pass
                            break

            # --------------------- Optical Flow как fallback ---------------------
            if not detected and prev_gray is not None and prev_point is not None and gray is not None:
                try:
                    new_pt, st, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_point, None,
                        winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    if st is not None and st.shape[0] > 0 and st[0][0] == 1:
                        nx, ny = float(new_pt.reshape(2)[0]), float(new_pt.reshape(2)[1])
                        measure = np.array([[nx], [ny]], dtype=np.float32)
                        detected = True
                except Exception:
                    pass

            # --------------------- Инициализация Kalman ---------------------
            if not kalman_init and detected and measure is not None:
                kalman = cv2.KalmanFilter(4, 2)
                kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
                kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                   [0, 1, 0, 1],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1]], dtype=np.float32)
                kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
                kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
                kalman.errorCovPost = np.eye(4, dtype=np.float32)
                kalman.statePost = np.array([[measure[0, 0]],
                                            [measure[1, 0]],
                                            [0.0], [0.0]], dtype=np.float32)
                kalman_init = True
                tracked = (int(measure[0, 0]), int(measure[1, 0]))
                prev_point = np.array([[[tracked[0], tracked[1]]]], dtype=np.float32)
                lost = 0

            else:
                # --------------------- Обновление/предсказание Kalman ---------------------
                if kalman_init:
                    try:
                        pred = kalman.predict()
                    except Exception as e:
                        pred = None
                        self.log(f"[WARN] Kalman predict error: {e}")

                    if detected and measure is not None:
                        try:
                            kalman.correct(measure)
                        except Exception:
                            pass
                        tracked = (int(measure[0, 0]), int(measure[1, 0]))
                        prev_point = np.array([[[tracked[0], tracked[1]]]], dtype=np.float32)
                        lost = 0
                    else:
                        lost += 1
                        if lost < MAX_LOST and pred is not None:
                            tx = int(pred[0, 0])
                            ty = int(pred[1, 0])
                            tracked = (tx, ty)
                            prev_point = np.array([[[tx, ty]]], dtype=np.float32)
                        else:
                            # Потеряли объект
                            kalman_init = False
                            tracked = None
                else:
                    tracked = None

            # Обновляем предыдущее серое изображение
            if gray is not None:
                prev_gray = gray.copy()

            # --------------------- Аннотация кадра ---------------------
            try:
                annotated = res.plot()
                if annotated is None:
                    annotated = frame.copy()
            except Exception:
                annotated = frame.copy()

            if tracked is not None:
                try:
                    x, y = tracked
                    cv2.line(annotated, (x, y - 16), (x, y + 16), (0, 0, 255), 2)
                    cv2.line(annotated, (x - 16, y), (x + 16, y), (0, 0, 255), 2)
                    cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
                    if detected_class is not None:
                        lbl = CLASS_NAMES.get(detected_class, "TRACKED")
                        cv2.putText(annotated, lbl, (x + 8, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except Exception:
                    pass

            # Конвертация в RGB и отправка в GUI
            try:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = annotated[..., ::-1]

            try:
                self.frame_ready.emit(rgb)
            except Exception:
                pass

            # Логирование FPS
            frame_count += 1
            now = time.time()
            if now - t_last >= 1.0:
                self.log(f"[FPS] {frame_count} | {w}x{h}")
                frame_count = 0
                t_last = now

        # --------------------- Очистка ---------------------
        try:
            cap.release()
        except Exception:
            pass
        self.log("[INFO] Worker finished")

# ==================================================================
# Графический интерфейс (PyQt6)
# ==================================================================
class PQ6App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PQ6 Drone Detector")
        self.resize(1400, 900)

        self.worker = None
        self.sound = SoundPlayer()
        self.status_q = queue.Queue()

        layout = QtWidgets.QHBoxLayout(self)

        # --------------------- Левая панель управления ---------------------
        ctrl = QtWidgets.QVBoxLayout()

        # Выбор модели
        btn_model = QtWidgets.QPushButton("Choose model (.pt)")
        btn_model.clicked.connect(self.choose_model)
        self.lbl_model = QtWidgets.QLabel("Model: -")
        ctrl.addWidget(btn_model)
        ctrl.addWidget(self.lbl_model)

        # Выбор видеофайла
        btn_video = QtWidgets.QPushButton("Choose video")
        btn_video.clicked.connect(self.choose_video)
        self.lbl_video = QtWidgets.QLabel("Source: -")
        ctrl.addWidget(btn_video)
        ctrl.addWidget(self.lbl_video)

        # Камера
        cam_h = QtWidgets.QHBoxLayout()
        self.cam_index = QtWidgets.QSpinBox()
        self.cam_index.setRange(0, 10)
        btn_cam = QtWidgets.QPushButton("Use camera")
        btn_cam.clicked.connect(self.use_camera)
        cam_h.addWidget(QtWidgets.QLabel("Cam idx:"))
        cam_h.addWidget(self.cam_index)
        cam_h.addWidget(btn_cam)
        ctrl.addLayout(cam_h)

        # Порог уверенности
        ctrl.addWidget(QtWidgets.QLabel("Confidence"))
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(30)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        self.lbl_conf = QtWidgets.QLabel("0.30")
        ctrl.addWidget(self.conf_slider)
        ctrl.addWidget(self.lbl_conf)

        # Выбор классов для отслеживания
        ctrl.addWidget(QtWidgets.QLabel("Track classes:"))
        self.class_checks = {}
        for cid, nm in CLASS_NAMES.items():
            cb = QtWidgets.QCheckBox(nm)
            cb.setChecked(True)
            ctrl.addWidget(cb)
            self.class_checks[cid] = cb

        # Выбор звукового файла
        btn_sound = QtWidgets.QPushButton("Choose sound file")
        btn_sound.clicked.connect(self.choose_sound)
        self.lbl_sound = QtWidgets.QLabel("Sound: -")
        ctrl.addWidget(btn_sound)
        ctrl.addWidget(self.lbl_sound)

        # Громкость
        ctrl.addWidget(QtWidgets.QLabel("Volume"))
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(100)
        self.vol_slider.valueChanged.connect(self.on_volume_change)
        ctrl.addWidget(self.vol_slider)

        # Кнопка Start/Stop
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self.start_stop)
        ctrl.addWidget(self.btn_start)

        # Лог
        ctrl.addWidget(QtWidgets.QLabel("Log"))
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(1000)
        ctrl.addWidget(self.log_edit)

        ctrl.addStretch()
        layout.addLayout(ctrl, 0)

        # --------------------- Правая панель — видео ---------------------
        vbox = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(1280, 720)
        vbox.addWidget(self.video_label, 1)
        self.res_label = QtWidgets.QLabel("Resolution: -")
        vbox.addWidget(self.res_label)
        layout.addLayout(vbox, 1)

        # Таймер для опроса очереди статусов
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll_status)
        self.timer.start()

        # Внутреннее состояние
        self.model_path = None
        self.source_spec = {"type": None, "value": None}
        self.conf_value = 0.30

    # --------------------- Вспомогательные методы ---------------------
    def log(self, msg: str):
        """Добавляет сообщение в лог с меткой времени."""
        ts = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {msg}")

    def poll_status(self):
        """Опрашивает очередь статусов от рабочего потока."""
        try:
            while True:
                msg = self.status_q.get_nowait()
                self.log(msg)
        except queue.Empty:
            pass

    # --------------------- Обработчики интерфейса ---------------------
    def choose_model(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose model (.pt)", "", "PT Files (*.pt);;All Files (*)")
        if p:
            self.model_path = p
            self.lbl_model.setText(f"Model: {p}")
            self.log(f"[INFO] Model selected: {p}")

    def choose_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose video", "", "Video (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if p:
            abs_p = str(Path(p).resolve())
            sp = get_short_path(abs_p)
            self.source_spec = {"type": "file", "value": sp}
            self.lbl_video.setText(f"File: {abs_p}")
            self.log(f"[INFO] Video selected: {abs_p}")

    def use_camera(self):
        idx = int(self.cam_index.value())
        self.source_spec = {"type": "camera", "value": idx}
        self.lbl_video.setText(f"Camera: {idx}")
        self.log(f"[INFO] Using camera index {idx}")

    def on_conf_change(self, v):
        self.conf_value = float(v) / 100.0
        self.lbl_conf.setText(f"{self.conf_value:.2f}")

    def choose_sound(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose sound", "", "Sound (*.wav *.mp3 *.ogg);;All Files (*)")
        if p:
            ok, err = self.sound.load(p)
            if ok:
                self.lbl_sound.setText(f"Sound: {p}")
                self.log("[INFO] Sound loaded")
            else:
                self.log(f"[ERROR] Sound load failed: {err}")

    def on_volume_change(self, v):
        vol = float(v) / 100.0
        self.sound.set_volume(vol)
        self.log(f"[INFO] Volume set to {vol:.2f}")

    def start_stop(self):
        """Запуск или остановка обработки."""
        # Остановка
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
            self.btn_start.setText("Start")
            self.log("[INFO] Worker stopped by user")
            return

        # Проверки перед запуском
        if not self.model_path:
            self.log("[ERROR] Select model first")
            return
        if self.source_spec["type"] is None or self.source_spec["value"] is None:
            self.log("[ERROR] Select video file or camera")
            return

        # Список выбранных классов
        sel = {cid: cb.isChecked() for cid, cb in self.class_checks.items()}

        # Создание и запуск рабочего потока
        self.worker = WorkerThread(self.model_path, self.source_spec, self.conf_value, sel, self.sound)
        self.worker.frame_ready.connect(self.update_frame, QtCore.Qt.ConnectionType.QueuedConnection)
        self.worker.status.connect(self.log)
        self.worker.start()
        self.btn_start.setText("Stop")
        self.log("[INFO] Worker started")

    @QtCore.pyqtSlot(np.ndarray)
    def update_frame(self, arr: np.ndarray):
        """Обновление изображения в интерфейсе."""
        try:
            h, w = arr.shape[:2]
            self.res_label.setText(f"Resolution: {w}x{h}")
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        except Exception as e:
            self.log(f"[ERROR] update_frame: {e}")

# ==================================================================
# Точка входа
# ==================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PQ6App()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
