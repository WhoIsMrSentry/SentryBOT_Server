from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import subprocess
import tempfile
import winsound
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from config import settings

logger = logging.getLogger("qt_gui")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if not settings.xtts_use_gpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@dataclass
class RobotEndpoints:
    host: str
    gateway_port: int = 8080
    camera_port: int = 8000

    def gateway_base(self) -> str:
        return f"http://{self.host}:{self.gateway_port}"

    def camera_base(self) -> str:
        return f"http://{self.host}:{self.camera_port}"


@dataclass
class ServerEndpoints:
    host: str
    port: int

    def base(self) -> str:
        return f"http://{self.host}:{self.port}"


class WorkerSignals(QtCore.QObject):
    done = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)


class HttpWorker(QtCore.QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            res = self.fn()
            self.signals.done.emit(res)
        except Exception as exc:
            self.signals.error.emit(str(exc))


class CameraWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int | None = None):
        super().__init__()
        self._running = False
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None

    @QtCore.pyqtSlot()
    def start(self):
        try:
            import cv2
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                self._cap = cv2.VideoCapture(self.camera_index, backend)
                if self._cap.isOpened():
                    break
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

            if self._cap is None or not self._cap.isOpened():
                self.error.emit("PC kamera açılamadı (backend denemeleri başarısız)")
                return
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            if self.fps:
                self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self._running = True
            while self._running:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    self.frame_ready.emit(frame)
                if self.fps:
                    delay_ms = max(1, int(1000 / max(self.fps, 1)))
                    QtCore.QThread.msleep(delay_ms)
                else:
                    QtCore.QThread.msleep(1)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None

    def stop(self):
        self._running = False


class VisionWorker(QtCore.QObject):
    results_ready = QtCore.pyqtSignal(dict, str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, frame_getter: Callable[[], Any]):
        super().__init__()
        self._running = False
        self._lock = threading.Lock()
        self._mode = "none"
        self._use_local = True
        self._stream_url = ""
        self._frame_getter = frame_getter
        self._svc = None

    def set_mode(self, mode: str):
        with self._lock:
            self._mode = mode

    def set_source(self, use_local: bool, stream_url: str):
        with self._lock:
            self._use_local = use_local
            self._stream_url = stream_url or ""

    def stop(self):
        self._running = False

    @QtCore.pyqtSlot()
    def start(self):
        self._running = True
        try:
            while self._running:
                with self._lock:
                    mode = self._mode
                    use_local = self._use_local
                    stream_url = self._stream_url

                if mode == "none":
                    QtCore.QThread.msleep(50)
                    continue

                image_bytes = None
                if use_local:
                    frame = self._frame_getter()
                    if frame is not None:
                        import cv2
                        ok, buf = cv2.imencode(".jpg", frame)
                        if ok:
                            image_bytes = buf.tobytes()
                else:
                    if not stream_url:
                        QtCore.QThread.msleep(50)
                        continue
                    try:
                        import requests
                        resp = requests.get(stream_url, timeout=5)
                        if resp.status_code == 200:
                            image_bytes = resp.content
                    except Exception:
                        pass

                if not image_bytes:
                    QtCore.QThread.msleep(10)
                    continue

                if self._svc is None:
                    from services.vision_service import VisionService
                    self._svc = VisionService()
                    self._svc.initialize()

                try:
                    results = self._svc.process_image(image_bytes, mode=mode)
                    self.results_ready.emit({"ok": True, "results": results, "local": True}, mode)
                except Exception as exc:
                    self.error.emit(str(exc))

                QtCore.QThread.msleep(50)
        except Exception as exc:
            self.error.emit(str(exc))


class RobotControlWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Control GUI v2")
        self.setMinimumSize(1500, 860)
        self.thread_pool = QtCore.QThreadPool.globalInstance()

        self.server_endpoints = ServerEndpoints(host="127.0.0.1", port=5000)
        self.robot_endpoints = RobotEndpoints(host="127.0.0.1")

        self._stream_timer = QtCore.QTimer(self)
        self._stream_timer.setInterval(200)
        self._stream_timer.timeout.connect(self._pull_snapshot)

        self._analysis_timer = QtCore.QTimer(self)
        self._analysis_timer.setInterval(1000)
        self._analysis_timer.timeout.connect(self._process_snapshot)
        self._analysis_busy = False

        self._ssh_lock = threading.Lock()
        self._ssh_client = None

        self._robot_connected = False
        self._camera_thread: Optional[QtCore.QThread] = None
        self._camera_worker: Optional[CameraWorker] = None
        self._camera_lock = threading.Lock()
        self._camera_frame = None
        self._local_preview_active = False
        self._analysis_active = False
        self._analysis_thread: Optional[QtCore.QThread] = None
        self._analysis_worker: Optional[VisionWorker] = None

        self._last_structured: Optional[dict] = None
        self._last_vision_results: Optional[dict] = None
        self._last_vision_mode: Optional[str] = None
        self._local_vision_service = None

        self._piper_model = None
        self._piper_model_path = None
        self._xtts_model = None
        self._tts_busy = False
        self._xtts_loading = False

        self._build_ui()
        self._apply_style()
        self._load_piper_voices()
        self._update_tts_ui()
        self._refresh_personas()
        self._maybe_preload_xtts()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # Left Panel (Controls)
        self.left_panel = QtWidgets.QFrame()
        self.left_panel.setObjectName("panel")
        left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        left_layout.setSpacing(10)
        self.left_panel.setMinimumWidth(460)
        self.left_panel.setMaximumWidth(520)

        left_layout.addWidget(self._build_top_status())
        left_layout.addWidget(self._build_section_buttons())

        self.section_stack = QtWidgets.QStackedWidget()
        self.section_stack.addWidget(self._build_robot_actions())
        self.section_stack.addWidget(self._build_vision_controls())
        self.section_stack.addWidget(self._build_audio_controls())
        self.section_stack.addWidget(self._build_llm_controls())
        left_layout.addWidget(self.section_stack)

        # Center Video Panel
        self.video_panel = QtWidgets.QFrame()
        self.video_panel.setObjectName("panel")
        video_layout = QtWidgets.QVBoxLayout(self.video_panel)
        video_layout.setSpacing(10)
        self.video_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self.video_label = QtWidgets.QLabel("Video Stream Disconnected")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(520)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background: #0c0507; color: #c9a3ac; border-radius: 12px;")
        video_layout.addWidget(self.video_label)

        self.stream_controls = QtWidgets.QHBoxLayout()
        self.stream_url = QtWidgets.QLineEdit()
        self.stream_url.setPlaceholderText("Snapshot URL (ör: http://robot:8000/camera/snap)")
        self.stream_start_btn = QtWidgets.QPushButton("Stream Başlat")
        self.stream_stop_btn = QtWidgets.QPushButton("Stream Durdur")
        self.stream_start_btn.clicked.connect(self._start_stream)
        self.stream_stop_btn.clicked.connect(self._stop_stream)
        self.stream_controls.addWidget(self.stream_url)
        self.stream_controls.addWidget(self.stream_start_btn)
        self.stream_controls.addWidget(self.stream_stop_btn)
        video_layout.addLayout(self.stream_controls)

        # Right Panel (Outputs & Terminal)
        self.right_panel = QtWidgets.QFrame()
        self.right_panel.setObjectName("panel")
        right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        right_layout.setSpacing(10)
        self.right_panel.setMinimumWidth(360)
        self.right_panel.setMaximumWidth(420)

        right_layout.addWidget(self._build_llm_outputs())
        right_layout.addWidget(self._build_terminal_panel())
        right_layout.addWidget(self._build_log_panel())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.video_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([520, 900, 380])

        root_layout.addWidget(splitter)

    def _build_top_status(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        title = QtWidgets.QLabel("Robot Durum")
        title.setStyleSheet("font-weight: 600;")

        self.robot_conn_label = QtWidgets.QLabel("Disconnected")
        self.robot_state_label = QtWidgets.QLabel("Unknown")
        self.robot_emotion_label = QtWidgets.QLabel("Unknown")
        self.robot_persona_label = QtWidgets.QLabel("Unknown")

        self.robot_ip = QtWidgets.QLineEdit("127.0.0.1")
        self.server_host = QtWidgets.QLineEdit("127.0.0.1")
        self.server_port = QtWidgets.QLineEdit("5000")
        self.connect_btn = QtWidgets.QPushButton("Robotu Bağla")
        self.connect_btn.clicked.connect(self._connect_robot)

        layout.addWidget(title, 0, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Connection"), 1, 0)
        layout.addWidget(self.robot_conn_label, 1, 1)
        layout.addWidget(QtWidgets.QLabel("State"), 2, 0)
        layout.addWidget(self.robot_state_label, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Emotions"), 3, 0)
        layout.addWidget(self.robot_emotion_label, 3, 1)
        layout.addWidget(QtWidgets.QLabel("Persona"), 4, 0)
        layout.addWidget(self.robot_persona_label, 4, 1)

        layout.addWidget(QtWidgets.QLabel("Robot IP"), 5, 0)
        layout.addWidget(self.robot_ip, 5, 1)
        layout.addWidget(QtWidgets.QLabel("Server Host"), 6, 0)
        layout.addWidget(self.server_host, 6, 1)
        layout.addWidget(QtWidgets.QLabel("Server Port"), 7, 0)
        layout.addWidget(self.server_port, 7, 1)
        layout.addWidget(self.connect_btn, 8, 0, 1, 2)

        return widget

    def _build_section_buttons(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn_actions = QtWidgets.QPushButton("Actions")
        self.btn_vision = QtWidgets.QPushButton("Vision")
        self.btn_audio = QtWidgets.QPushButton("TTS")
        self.btn_llm = QtWidgets.QPushButton("LLM")

        self.btn_actions.clicked.connect(lambda: self._switch_section(0))
        self.btn_vision.clicked.connect(lambda: self._switch_section(1))
        self.btn_audio.clicked.connect(lambda: self._switch_section(2))
        self.btn_llm.clicked.connect(lambda: self._switch_section(3))

        layout.addWidget(self.btn_actions)
        layout.addWidget(self.btn_vision)
        layout.addWidget(self.btn_audio)
        layout.addWidget(self.btn_llm)
        return widget

    def _build_robot_actions(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Robot Actions")
        layout = QtWidgets.QFormLayout(box)

        self.effect_select = QtWidgets.QComboBox()
        self.effect_select.addItems(["COMET", "BREATHE", "RAINBOW", "BOUNCE", "CALM"])
        self.effect_btn = QtWidgets.QPushButton("Effect Gönder")
        self.effect_btn.clicked.connect(self._send_effect)

        self.emote_select = QtWidgets.QComboBox()
        self.emote_select.addItems(["neutral", "joy", "surprise", "sadness", "anger", "fear", "love"])
        self.emote_btn = QtWidgets.QPushButton("Emote Gönder")
        self.emote_btn.clicked.connect(self._send_emote)

        layout.addRow("Effect", self.effect_select)
        layout.addRow(self.effect_btn)
        layout.addRow("Emote", self.emote_select)
        layout.addRow(self.emote_btn)
        return box

    def _build_vision_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Vision")
        layout = QtWidgets.QVBoxLayout(box)

        self.use_local_camera = QtWidgets.QCheckBox("PC Kamera Kullan")
        self.use_local_camera.stateChanged.connect(self._toggle_local_camera)

        modes_layout = QtWidgets.QHBoxLayout()
        self.mode_select = QtWidgets.QComboBox()
        self.mode_select.addItems(["none", "object", "face", "hand", "attributes", "age_emotion", "motion", "finger", "face_recognize"])
        self.mode_select.setCurrentText("none")
        self.mode_select.currentTextChanged.connect(self._on_vision_mode_changed)
        modes_layout.addWidget(QtWidgets.QLabel("Mod"))
        modes_layout.addWidget(self.mode_select)

        buttons_layout = QtWidgets.QHBoxLayout()
        self.vision_snapshot_btn = QtWidgets.QPushButton("Snapshot İşle")
        self.vision_stream_btn = QtWidgets.QPushButton("Stream İşle")
        self.vision_stop_btn = QtWidgets.QPushButton("Stream Durdur")
        self.vision_snapshot_btn.clicked.connect(self._process_snapshot)
        self.vision_stream_btn.clicked.connect(self._start_stream_analysis)
        self.vision_stop_btn.clicked.connect(self._stop_stream_analysis)
        buttons_layout.addWidget(self.vision_snapshot_btn)
        buttons_layout.addWidget(self.vision_stream_btn)
        buttons_layout.addWidget(self.vision_stop_btn)

        self.vision_output = QtWidgets.QTextEdit()
        self.vision_output.setReadOnly(True)

        train_box = QtWidgets.QGroupBox("Yüz Tanıma Eğitimi")
        train_layout = QtWidgets.QFormLayout(train_box)
        self.face_name_input = QtWidgets.QLineEdit()
        self.face_name_input.setPlaceholderText("Kişi adı (ör: owner)")
        self.face_sample_count = QtWidgets.QSpinBox()
        self.face_sample_count.setMinimum(3)
        self.face_sample_count.setMaximum(100)
        self.face_sample_count.setValue(10)
        self.face_collect_btn = QtWidgets.QPushButton("Örnek Topla")
        self.face_collect_btn.clicked.connect(self._collect_face_samples)
        self.face_encode_btn = QtWidgets.QPushButton("Encoding Oluştur")
        self.face_encode_btn.clicked.connect(self._build_face_encodings)
        self.face_train_status = QtWidgets.QLabel("")

        train_layout.addRow("Kişi", self.face_name_input)
        train_layout.addRow("Örnek Sayısı", self.face_sample_count)
        train_layout.addRow(self.face_collect_btn)
        train_layout.addRow(self.face_encode_btn)
        train_layout.addRow("Durum", self.face_train_status)

        layout.addWidget(self.use_local_camera)
        layout.addLayout(modes_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(train_box)
        layout.addWidget(self.vision_output)
        return box

    def _build_audio_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Audio (TTS)")
        layout = QtWidgets.QFormLayout(box)

        self.tts_engine = QtWidgets.QComboBox()
        self.tts_engine.addItems(["piper", "xtts"])
        self.tts_engine.currentTextChanged.connect(self._update_tts_ui)

        self.tts_lang = QtWidgets.QComboBox()
        self.tts_lang.addItems(["tr", "en", "de", "es", "fr", "ru", "ar"])

        self.piper_voice = QtWidgets.QComboBox()
        self.xtts_voice_path = QtWidgets.QLineEdit("")
        self.xtts_browse_btn = QtWidgets.QPushButton("Ses Dosyası Seç")
        self.xtts_browse_btn.clicked.connect(self._pick_xtts_voice)

        self.xtts_preload_btn = QtWidgets.QPushButton("XTTS Hazırla")
        self.xtts_preload_btn.clicked.connect(self._preload_xtts)
        self.xtts_stop_btn = QtWidgets.QPushButton("XTTS Durdur")
        self.xtts_stop_btn.clicked.connect(self._stop_xtts)

        self.tts_text = QtWidgets.QTextEdit()
        self.tts_speak_btn = QtWidgets.QPushButton("Sentezle")
        self.tts_speak_btn.clicked.connect(self._tts_speak)
        self.tts_play_on_robot = QtWidgets.QCheckBox("Robotta çal")

        self.tts_status = QtWidgets.QLabel("")

        self.tts_lang_label = QtWidgets.QLabel("Dil")
        self.piper_voice_label = QtWidgets.QLabel("Piper Voice")
        self.xtts_voice_label = QtWidgets.QLabel("XTTS Voice")

        self.xtts_voice_row = QtWidgets.QWidget()
        xtts_row = QtWidgets.QHBoxLayout(self.xtts_voice_row)
        xtts_row.setContentsMargins(0, 0, 0, 0)
        xtts_row.addWidget(self.xtts_voice_path)
        xtts_row.addWidget(self.xtts_browse_btn)

        self.xtts_action_row = QtWidgets.QWidget()
        xtts_actions = QtWidgets.QHBoxLayout(self.xtts_action_row)
        xtts_actions.setContentsMargins(0, 0, 0, 0)
        xtts_actions.addWidget(self.xtts_preload_btn)
        xtts_actions.addWidget(self.xtts_stop_btn)

        layout.addRow("Engine", self.tts_engine)
        layout.addRow(self.tts_lang_label, self.tts_lang)
        layout.addRow(self.piper_voice_label, self.piper_voice)
        layout.addRow(self.xtts_voice_label, self.xtts_voice_row)
        layout.addRow("XTTS", self.xtts_action_row)

        layout.addRow("Metin", self.tts_text)
        layout.addRow(self.tts_speak_btn)
        layout.addRow(self.tts_play_on_robot)
        layout.addRow("Durum", self.tts_status)

        return box

    def _build_llm_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("LLM")
        layout = QtWidgets.QFormLayout(box)

        self.llm_source = QtWidgets.QComboBox()
        self.llm_source.addItems(["server", "robot-ollama"])

        self.llm_model = QtWidgets.QComboBox()
        self.llm_refresh_btn = QtWidgets.QPushButton("Modelleri Yenile")
        self.llm_refresh_btn.clicked.connect(self._refresh_models)

        self.persona_select = QtWidgets.QComboBox()
        self.persona_refresh_btn = QtWidgets.QPushButton("Personaları Yenile")
        self.persona_refresh_btn.clicked.connect(self._refresh_personas)
        self.persona_select.currentTextChanged.connect(self._select_persona)

        self.llm_structured = QtWidgets.QCheckBox("Structured Output")
        self.llm_speak = QtWidgets.QCheckBox("LLM çıktısını okut")

        self.stt_auto_tts = QtWidgets.QCheckBox("STT diline göre TTS")
        self.stt_listen_btn = QtWidgets.QPushButton("STT Dinle")
        self.stt_stop_btn = QtWidgets.QPushButton("STT Durdur")
        self.stt_listen_btn.clicked.connect(self._stt_start)
        self.stt_stop_btn.clicked.connect(self._stt_stop)
        self.stt_last = QtWidgets.QLabel("STT: -")

        self.llm_system = QtWidgets.QTextEdit()
        self.llm_system.setPlaceholderText("System Prompt")
        self.llm_input = QtWidgets.QTextEdit()
        self.llm_send_btn = QtWidgets.QPushButton("Gönder")
        self.llm_send_btn.clicked.connect(self._send_llm)

        layout.addRow("Kaynak", self.llm_source)
        layout.addRow("Model", self.llm_model)
        layout.addRow(self.llm_refresh_btn)
        layout.addRow("Persona", self.persona_select)
        layout.addRow(self.persona_refresh_btn)
        layout.addRow(self.llm_structured)
        layout.addRow(self.llm_speak)
        layout.addRow(self.stt_auto_tts)
        layout.addRow(self.stt_last)
        stt_row = QtWidgets.QHBoxLayout()
        stt_row.addWidget(self.stt_listen_btn)
        stt_row.addWidget(self.stt_stop_btn)
        stt_row_widget = QtWidgets.QWidget()
        stt_row_widget.setLayout(stt_row)
        layout.addRow(stt_row_widget)
        layout.addRow("System", self.llm_system)
        layout.addRow("Input", self.llm_input)
        layout.addRow(self.llm_send_btn)
        return box

    def _build_llm_outputs(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("LLM Outputs")
        layout = QtWidgets.QVBoxLayout(box)

        self.llm_output = QtWidgets.QTextEdit()
        self.llm_output.setReadOnly(True)
        self.llm_output.setFixedHeight(160)

        self.structured_output = QtWidgets.QTextEdit()
        self.structured_output.setReadOnly(True)
        self.structured_output.setFixedHeight(260)

        self.apply_actions_btn = QtWidgets.QPushButton("Structured Actions Uygula")
        self.apply_actions_btn.clicked.connect(self._apply_structured_actions)

        layout.addWidget(QtWidgets.QLabel("LLM Çıktısı"))
        layout.addWidget(self.llm_output)
        layout.addWidget(QtWidgets.QLabel("Structured Output"))
        layout.addWidget(self.structured_output)
        layout.addWidget(self.apply_actions_btn)
        return box

    def _build_terminal_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("SSH Terminal")
        layout = QtWidgets.QVBoxLayout(box)

        self.terminal_output = QtWidgets.QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_input = QtWidgets.QLineEdit()
        self.terminal_input.setPlaceholderText("ssh sentrybot@192.168.25.54")
        self.terminal_input.returnPressed.connect(self._terminal_send)

        layout.addWidget(self.terminal_output)
        layout.addWidget(self.terminal_input)
        return box

    def _build_log_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Log")
        layout = QtWidgets.QVBoxLayout(box)
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(140)
        layout.addWidget(self.log_output)
        return box

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget { background-color: #3b000a; color: #f3d7dc; font-family: Segoe UI; }
            QGroupBox { border: 1px solid #7a0f1f; border-radius: 8px; margin-top: 8px; padding: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLineEdit, QTextEdit, QComboBox { background: #5a0011; border: 1px solid #8c1b2e; border-radius: 6px; padding: 6px; }
            QPushButton { background: #88001b; border: 1px solid #a81c35; border-radius: 6px; padding: 6px; }
            QPushButton:hover { background: #a0001f; }
            QCheckBox { padding: 4px; }
            """
        )

    def _switch_section(self, index: int):
        if hasattr(self, "section_stack"):
            self.section_stack.setCurrentIndex(index)

    def _on_vision_mode_changed(self, mode: str):
        if self._analysis_active:
            self._stop_stream_analysis()
            self._log("İşleme durduruldu (mod değişti)")
        self._update_camera_state()
        if mode == "none":
            self.vision_output.setPlainText("İşleme kapalı (none)")

    # ---------------- Helpers ----------------
    def _log(self, msg: str):
        self.log_output.append(msg)

    def _connect_robot(self):
        self.robot_endpoints = RobotEndpoints(host=self.robot_ip.text().strip())
        self.server_endpoints = ServerEndpoints(
            host=self.server_host.text().strip(),
            port=int(self.server_port.text()),
        )
        self.stream_url.setText(f"{self.robot_endpoints.camera_base()}/camera/snap")
        self._log("Robot endpointleri güncellendi.")
        self._robot_connected = False
        self._configure_vision_push()
        self._fetch_robot_status()
        self._refresh_models()

    def _fetch_robot_status(self):
        def _do():
            res = requests.get(f"{self.robot_endpoints.gateway_base()}/status", timeout=2)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(self._update_robot_status)
        worker.signals.error.connect(lambda e: self._on_gateway_error(e))
        self.thread_pool.start(worker)

    def _update_robot_status(self, payload: dict):
        self.robot_conn_label.setText("Connected")
        self._robot_connected = True
        configured = payload.get("configured", {})
        self.robot_state_label.setText("ok" if payload.get("ok") else "unknown")
        self.robot_persona_label.setText("ollama" if configured.get("ollama") else "-")
        self._fetch_state_manager()

    def _fetch_state_manager(self):
        def _do():
            res = requests.get(f"{self.robot_endpoints.gateway_base()}/state/get", timeout=2)
            if res.status_code != 200:
                return None
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(self._update_state_manager)
        worker.signals.error.connect(lambda _: None)
        self.thread_pool.start(worker)

    def _update_state_manager(self, payload: Optional[dict]):
        if not payload:
            return
        self.robot_state_label.setText(str(payload.get("operational", "unknown")))
        emotions = payload.get("emotions")
        if isinstance(emotions, list):
            self.robot_emotion_label.setText(", ".join(emotions[:4]))

    def _on_gateway_error(self, err: str):
        self._robot_connected = False
        self.robot_conn_label.setText("Disconnected")
        self._log(f"Gateway status error: {err}")

    def _send_effect(self):
        name = self.effect_select.currentText()

        def _do():
            res = requests.post(
                f"{self.robot_endpoints.gateway_base()}/interactions/effect",
                json={"name": name, "duration_ms": 900},
                timeout=5,
            )
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda _: self._log(f"Effect gönderildi: {name}"))
        worker.signals.error.connect(lambda e: self._log(f"Effect error: {e}"))
        self.thread_pool.start(worker)

    def _send_emote(self):
        emotion = self.emote_select.currentText()

        def _do():
            res = requests.post(
                f"{self.robot_endpoints.gateway_base()}/neopixel/emote",
                params={"emotions": emotion},
                timeout=5,
            )
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda _: self._log(f"Emote gönderildi: {emotion}"))
        worker.signals.error.connect(lambda e: self._log(f"Emote error: {e}"))
        self.thread_pool.start(worker)

    def _refresh_models(self):
        def _do():
            res = requests.get(f"{self.server_endpoints.base()}/llm/health", timeout=4)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(self._update_models)
        worker.signals.error.connect(lambda e: self._log(f"Model refresh error: {e}"))
        self.thread_pool.start(worker)

    def _update_models(self, payload: dict):
        models = payload.get("models", [])
        self.llm_model.clear()
        self.llm_model.addItems(models)
        if models:
            self.llm_model.setCurrentIndex(0)

    def _configure_vision_push(self):
        payload = {
            "robot_gateway_url": self.robot_endpoints.gateway_base(),
        }

        def _do():
            res = requests.post(f"{self.server_endpoints.base()}/vision/config", params=payload, timeout=3)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda _: self._log("Vision push ayarlandı"))
        worker.signals.error.connect(lambda e: self._log(f"Vision push error: {e}"))
        self.thread_pool.start(worker)

    # ---------------- Video ----------------
    def _toggle_local_camera(self):
        self._update_camera_state()

    def _camera_should_run(self) -> bool:
        if self._local_preview_active:
            return True
        stream_url = self.stream_url.text().strip() if hasattr(self, "stream_url") else ""
        use_local = self.use_local_camera.isChecked() or not self._robot_connected or not stream_url
        if self._analysis_active:
            return use_local
        if self._stream_timer.isActive():
            return self.use_local_camera.isChecked() or not stream_url
        return False

    def _update_camera_state(self):
        if self._camera_should_run():
            self._start_local_camera()
        else:
            self._stop_local_camera()

    def _start_local_camera(self):
        if self._camera_thread is not None:
            return
        self._camera_thread = QtCore.QThread(self)
        self._camera_worker = CameraWorker()
        self._camera_worker.moveToThread(self._camera_thread)
        self._camera_worker.frame_ready.connect(self._on_camera_frame)
        self._camera_worker.error.connect(self._on_camera_error)
        self._camera_thread.started.connect(self._camera_worker.start)
        self._camera_thread.start()

    def _stop_local_camera(self):
        if self._camera_worker is not None:
            self._camera_worker.stop()
        if self._camera_thread is not None:
            self._camera_thread.quit()
            self._camera_thread.wait(1500)
        self._camera_thread = None
        self._camera_worker = None
        with self._camera_lock:
            self._camera_frame = None

    def _on_camera_frame(self, frame):
        try:
            with self._camera_lock:
                self._camera_frame = frame
            if self._local_preview_active or self._analysis_active:
                self._render_local_frame(frame)
        except Exception:
            pass

    def _on_camera_error(self, err: str):
        self._log(f"Kamera hata: {err}")

    def _render_local_frame(self, frame):
        self._apply_vision_overlay(frame)
        image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
        )

    def _start_stream(self):
        stream_url = self.stream_url.text().strip() if hasattr(self, "stream_url") else ""
        if self.use_local_camera.isChecked() or not self._robot_connected or not stream_url:
            self._local_preview_active = True
            self._update_camera_state()
            self._stream_timer.stop()
        else:
            self._stream_timer.start()
        self._log("Stream başlatıldı.")

    def _stop_stream(self):
        self._local_preview_active = False
        self._stream_timer.stop()
        self._update_camera_state()
        self._log("Stream durduruldu.")

    def _pull_snapshot(self):
        if self.use_local_camera.isChecked() or not self._robot_connected:
            return

        url = self.stream_url.text().strip()
        if not url:
            return
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code != 200:
                return
            image = QtGui.QImage.fromData(resp.content)
            if image.isNull():
                return
            pix = QtGui.QPixmap.fromImage(image)
            self.video_label.setPixmap(
                pix.scaled(
                    self.video_label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.FastTransformation,
                )
            )
        except Exception:
            pass

    # ---------------- Vision ----------------
    def _process_snapshot(self):
        if self._analysis_busy:
            return
        mode = self.mode_select.currentText()
        if mode == "none":
            self.vision_output.setPlainText("İşleme kapalı (none)")
            return
        self._analysis_busy = True
        start_for_snapshot = False

        stream_url = self.stream_url.text().strip() if hasattr(self, "stream_url") else ""
        use_local = self.use_local_camera.isChecked() or not self._robot_connected or not stream_url

        if use_local:
            if self._camera_thread is None:
                self._start_local_camera()
                start_for_snapshot = True

        def _do():
            if use_local:
                frame = self._read_local_frame_wait(700)
                if frame is None:
                    raise RuntimeError("PC kamera okunamadı")
                import cv2
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError("PC kamera encode hata")
                image_bytes = buf.tobytes()
            else:
                img_resp = requests.get(stream_url, timeout=5)
                img_resp.raise_for_status()
                image_bytes = img_resp.content

            try:
                files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
                params = {"mode": mode}
                res = requests.post(
                    f"{self.server_endpoints.base()}/vision/process",
                    files=files,
                    params=params,
                    timeout=20,
                )
                res.raise_for_status()
                return res.json()
            except Exception:
                results = self._process_locally(image_bytes, mode)
                return results

        def _done(data):
            self._analysis_busy = False
            self.vision_output.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            self._last_vision_results = data
            self._last_vision_mode = mode
            if start_for_snapshot and not self._camera_should_run():
                self._stop_local_camera()
            self._update_camera_state()

        def _err(e):
            self._analysis_busy = False
            self.vision_output.setPlainText(f"Error: {e}")
            self._last_vision_results = None
            self._last_vision_mode = None
            if start_for_snapshot and not self._camera_should_run():
                self._stop_local_camera()
            self._update_camera_state()

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        worker.signals.error.connect(_err)
        self.thread_pool.start(worker)

    def _start_stream_analysis(self):
        if self.mode_select.currentText() == "none":
            self._log("İşleme kapalı (none). Sadece stream gösteriliyor.")
            self._start_stream()
            return
        self._analysis_active = True
        self._update_analysis_worker()
        self._update_camera_state()
        self._log("Vision stream analizi başladı.")

    def _stop_stream_analysis(self):
        self._analysis_active = False
        self._analysis_busy = False
        if self._analysis_worker is not None:
            self._analysis_worker.stop()
        if self._analysis_thread is not None:
            self._analysis_thread.quit()
            self._analysis_thread.wait(1500)
        self._analysis_worker = None
        self._analysis_thread = None
        self._update_camera_state()
        self._log("Vision stream analizi durdu.")

    def _read_local_frame(self):
        with self._camera_lock:
            if self._camera_frame is None:
                return None
            return self._camera_frame.copy()

    def _read_local_frame_wait(self, timeout_ms: int = 500):
        waited = 0
        step = 50
        while waited <= timeout_ms:
            frame = self._read_local_frame()
            if frame is not None:
                return frame
            QtCore.QThread.msleep(step)
            waited += step
        return None

    def _apply_vision_overlay(self, frame) -> None:
        if not self._last_vision_results or not self._last_vision_mode:
            return
        if self._last_vision_mode == "none":
            return
        try:
            import cv2
            h, w = frame.shape[:2]
            mode = self._last_vision_mode
            results = self._last_vision_results.get("results") or self._last_vision_results

            if mode == "object":
                for obj in results.get("objects", []):
                    bbox = obj.get("bbox") or []
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{obj.get('label', '')} {obj.get('confidence', 0):.2f}"
                        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            elif mode in ("face", "attributes"):
                for face in results.get("faces", []):
                    area = face.get("facial_area") or face.get("region") or {}
                    x = int(area.get("x", 0))
                    y = int(area.get("y", 0))
                    w0 = int(area.get("w", 0))
                    h0 = int(area.get("h", 0))
                    if w0 > 0 and h0 > 0:
                        cv2.rectangle(frame, (x, y), (x + w0, y + h0), (255, 0, 0), 2)

            elif mode == "age_emotion":
                for face in results.get("faces", []):
                    bbox = face.get("bbox") or []
                    if len(bbox) == 4:
                        x, y, w0, h0 = map(int, bbox)
                        cv2.rectangle(frame, (x, y), (x + w0, y + h0), (0, 200, 255), 2)
                        label = f"{face.get('age', '?')} | {face.get('emotion', '?')}"
                        cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            elif mode == "hand":
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17),
                ]
                for hand in results.get("hands", []):
                    landmarks = hand.get("landmarks") or []
                    pts = []
                    for lm in landmarks:
                        x = int((lm.get("x", 0.0)) * w)
                        y = int((lm.get("y", 0.0)) * h)
                        pts.append((x, y))
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                    for a, b in connections:
                        if a < len(pts) and b < len(pts):
                            cv2.line(frame, pts[a], pts[b], (0, 255, 255), 1)

            elif mode == "motion":
                motion = results.get("motion", {})
                for (x, y, w0, h0) in motion.get("areas", []):
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w0), int(y + h0)), (0, 255, 0), 2)

            elif mode == "finger":
                gesture = results.get("gesture", {})
                hands = gesture.get("hands", [])
                finger_tip_ids = [4, 8, 12, 16, 20]
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17),
                ]
                colors = [
                    (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255),
                    (0, 165, 255), (203, 192, 255), (255, 255, 0), (255, 255, 255), (0, 255, 127),
                ]
                for hand_idx, hand in enumerate(hands[:2]):
                    landmarks = hand.get("landmarks") or []
                    state = hand.get("state") or "00000"
                    pts = []
                    for lm in landmarks:
                        x = int(lm.get("x", 0.0) * w)
                        y = int(lm.get("y", 0.0) * h)
                        pts.append((x, y))
                    for a, b in connections:
                        if a < len(pts) and b < len(pts):
                            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                    for i, tip_id in enumerate(finger_tip_ids):
                        if i < len(state) and state[i] == "1" and tip_id < len(landmarks):
                            x, y = pts[tip_id]
                            color_idx = hand_idx * 5 + i
                            cv2.circle(frame, (x, y), 12, colors[color_idx], cv2.FILLED)
                        elif tip_id < len(landmarks):
                            x, y = pts[tip_id]
                            cv2.circle(frame, (x, y), 6, (90, 90, 90), cv2.FILLED)

            elif mode == "face_recognize":
                for face in results.get("faces", []):
                    bbox = face.get("bbox") or []
                    if len(bbox) == 4:
                        x, y, w0, h0 = map(int, bbox)
                        cv2.rectangle(frame, (x, y), (x + w0, y + h0), (255, 180, 0), 2)
                        name = face.get("name", "unknown")
                        cv2.putText(frame, name, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
        except Exception:
            pass

    # ---------------- Face Training ----------------
    def _collect_face_samples(self):
        name = self.face_name_input.text().strip()
        if not name:
            self.face_train_status.setText("Kişi adı boş")
            return
        count = int(self.face_sample_count.value())
        if count <= 0:
            self.face_train_status.setText("Örnek sayısı geçersiz")
            return

        base_dir = os.path.join(os.path.dirname(__file__), "models", "faces", name)
        os.makedirs(base_dir, exist_ok=True)

        start_for_capture = False
        if self.use_local_camera.isChecked() or not self._robot_connected:
            if self._camera_thread is None:
                self._start_local_camera()
                start_for_capture = True

        self.face_train_status.setText("Örnekler toplanıyor...")
        self.face_collect_btn.setEnabled(False)

        def _do():
            saved = 0
            for i in range(count):
                if self.use_local_camera.isChecked() or not self._robot_connected:
                    frame = self._read_local_frame_wait(1500)
                    if frame is None:
                        continue
                else:
                    url = self.stream_url.text().strip()
                    if not url:
                        raise RuntimeError("Snapshot URL boş")
                    img_resp = requests.get(url, timeout=5)
                    img_resp.raise_for_status()
                    data = np.frombuffer(img_resp.content, np.uint8)
                    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                filename = os.path.join(base_dir, f"{int(time.time()*1000)}_{i}.jpg")
                cv2.imwrite(filename, frame)
                saved += 1
                time.sleep(0.15)
            return saved

        def _done(saved: int):
            self.face_train_status.setText(f"{saved} örnek kaydedildi")
            self.face_collect_btn.setEnabled(True)
            if start_for_capture and not self._camera_should_run():
                self._stop_local_camera()

        def _err(e: str):
            self.face_train_status.setText(f"Hata: {e}")
            self.face_collect_btn.setEnabled(True)
            if start_for_capture and not self._camera_should_run():
                self._stop_local_camera()

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        worker.signals.error.connect(_err)
        self.thread_pool.start(worker)

    def _build_face_encodings(self):
        script_path = os.path.join(os.path.dirname(__file__), "tools", "build_face_encodings.py")
        if not os.path.exists(script_path):
            self.face_train_status.setText("Encoding script bulunamadı")
            return

        self.face_train_status.setText("Encoding oluşturuluyor...")
        self.face_encode_btn.setEnabled(False)

        def _do():
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__),
                timeout=180,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "Encoding hata")
            return proc.stdout.strip()

        def _done(msg: str):
            self.face_train_status.setText("Encoding tamamlandı")
            if msg:
                self._log(msg)
            self.face_encode_btn.setEnabled(True)

        def _err(e: str):
            self.face_train_status.setText(f"Hata: {e}")
            self.face_encode_btn.setEnabled(True)

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        worker.signals.error.connect(_err)
        self.thread_pool.start(worker)

    def _get_local_vision_service(self):
        if self._local_vision_service is None:
            from services.vision_service import VisionService
            svc = VisionService()
            svc.initialize()
            self._local_vision_service = svc
        return self._local_vision_service

    def _process_locally(self, image_bytes: bytes, mode: str) -> dict:
        svc = self._get_local_vision_service()
        results = svc.process_image(image_bytes, mode=mode)
        return {"ok": True, "results": results, "local": True}

    def _update_analysis_worker(self):
        mode = self.mode_select.currentText()
        stream_url = self.stream_url.text().strip() if hasattr(self, "stream_url") else ""
        use_local = self.use_local_camera.isChecked() or not self._robot_connected or not stream_url

        if self._analysis_worker is None:
            self._analysis_thread = QtCore.QThread(self)
            self._analysis_worker = VisionWorker(self._read_local_frame)
            self._analysis_worker.moveToThread(self._analysis_thread)
            self._analysis_worker.results_ready.connect(self._on_analysis_result)
            self._analysis_worker.error.connect(lambda e: self.vision_output.setPlainText(f"Error: {e}"))
            self._analysis_thread.started.connect(self._analysis_worker.start)
            self._analysis_thread.start()

        self._analysis_worker.set_mode(mode)
        self._analysis_worker.set_source(use_local, stream_url)

    def _on_analysis_result(self, data: dict, mode: str):
        self._last_vision_results = data
        self._last_vision_mode = mode
        self.vision_output.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))

    # ---------------- TTS ----------------
    def _load_piper_voices(self):
        base = os.path.join(os.path.dirname(__file__), "tts", "TTS", "PiperTTS")
        voices = []
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                for f in files:
                    if f.endswith(".onnx"):
                        voices.append(os.path.join(root, f))
        voices.sort()
        self.piper_voice.clear()
        for path in voices:
            name = os.path.relpath(path, base)
            self.piper_voice.addItem(name, path)

    def _update_tts_ui(self):
        engine = self.tts_engine.currentText()
        is_xtts = engine == "xtts"
        self.tts_lang.setVisible(not is_xtts)
        self.tts_lang_label.setVisible(not is_xtts)
        self.piper_voice.setVisible(not is_xtts)
        self.piper_voice_label.setVisible(not is_xtts)
        self.xtts_voice_label.setVisible(is_xtts)
        self.xtts_voice_row.setVisible(is_xtts)
        self.xtts_action_row.setVisible(is_xtts)

        if is_xtts and settings.xtts_preload_on_start:
            self._preload_xtts()

        self._apply_tts_config()

    def _pick_xtts_voice(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "XTTS Voice", "", "WAV Files (*.wav)")
        if path:
            self.xtts_voice_path.setText(path)
            self._apply_tts_config()

    def _apply_tts_config(self):
        engine = self.tts_engine.currentText()
        if engine == "piper":
            self.tts_status.setText("Piper hazır")
        else:
            self.tts_status.setText("XTTS hazır")

    def _maybe_preload_xtts(self):
        if not settings.xtts_preload_on_start:
            return
        if self.tts_engine.currentText() != "xtts":
            return
        self._preload_xtts()

    def _preload_xtts(self):
        if self._xtts_model is not None or self._xtts_loading:
            return
        self._xtts_loading = True
        self.tts_status.setText("XTTS yükleniyor...")
        self.xtts_preload_btn.setEnabled(False)

        def _do():
            return self._ensure_xtts_loaded()

        def _done(_):
            self._xtts_loading = False
            self.tts_status.setText("XTTS hazır")
            self.xtts_preload_btn.setEnabled(True)

        def _err(e: str):
            self._xtts_loading = False
            self.tts_status.setText(f"XTTS preload error: {e}")
            self.xtts_preload_btn.setEnabled(True)

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        worker.signals.error.connect(_err)
        self.thread_pool.start(worker)

    def _stop_xtts(self):
        if self._xtts_model is None:
            self.tts_status.setText("XTTS zaten durduruldu")
            return
        self._xtts_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.tts_status.setText("XTTS durduruldu")

    def _ensure_xtts_loaded(self):
        if self._xtts_model is None:
            from TTS.api import TTS
            try:
                import torch
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XttsAudioConfig
                from TTS.config.shared_configs import BaseDatasetConfig
                torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig])
            except Exception:
                pass
            self._xtts_model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=settings.xtts_use_gpu,
            )
        return True

    def _tts_speak(self):
        if self._tts_busy:
            self.tts_status.setText("TTS meşgul, lütfen bekleyin")
            return
        text = self.tts_text.toPlainText().strip()
        if not text:
            self.tts_status.setText("Metin boş")
            return

        engine = self.tts_engine.currentText()
        voice = None
        language = self.tts_lang.currentText() or "tr"
        if engine == "piper":
            voice = self.piper_voice.currentData()
        else:
            voice = self.xtts_voice_path.text().strip() or None

        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "engine": engine,
        }

        def _do():
            self._tts_busy = True
            self.tts_speak_btn.setEnabled(False)
            return self._synthesize_local(text, engine, voice, language)

        def _done(data: bytes):
            self.tts_status.setText(f"Ses üretildi ({len(data)} bytes)")
            if self.tts_play_on_robot.isChecked():
                self._play_on_robot(data)
            else:
                self._play_on_pc(data)
            self._tts_busy = False
            self.tts_speak_btn.setEnabled(True)

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        def _err(e: str):
            self.tts_status.setText(f"TTS error: {e}")
            self._tts_busy = False
            self.tts_speak_btn.setEnabled(True)

        worker.signals.error.connect(_err)
        self.thread_pool.start(worker)

    def _play_on_robot(self, wav_bytes: bytes):
        payload = {"data": base64.b64encode(wav_bytes).decode("utf-8")}

        def _do():
            res = requests.post(f"{self.robot_endpoints.gateway_base()}/speak/play", json=payload, timeout=10)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda _: self._log("Robotta oynatıldı."))
        worker.signals.error.connect(lambda e: self._log(f"Robot play error: {e}"))
        self.thread_pool.start(worker)

    def _play_on_pc(self, wav_bytes: bytes):
        try:
            import io
            import soundfile as sf
            import sounddevice as sd

            data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            sd.stop()
            sd.play(data, samplerate, blocking=False)
        except Exception as exc:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(wav_bytes)
                    tmp_path = tmp.name
                winsound.PlaySound(None, winsound.SND_PURGE)
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as exc2:
                self._log(f"PC play error: {exc}; winsound fallback error: {exc2}")

    # ---------------- LLM ----------------
    def _send_llm(self):
        query = self.llm_input.toPlainText().strip()
        if not query:
            return
        source = self.llm_source.currentText()
        structured = self.llm_structured.isChecked()

        def _do():
            if source == "robot-ollama":
                params = {"query": query, "structured": structured}
                res = requests.get(f"{self.robot_endpoints.gateway_base()}/ollama/chat", params=params, timeout=60)
                res.raise_for_status()
                return res.json()
            payload = {
                "query": query,
                "system": self.llm_system.toPlainText(),
                "model": self.llm_model.currentText() or None,
            }
            res = requests.post(f"{self.server_endpoints.base()}/llm/chat", json=payload, timeout=60)
            res.raise_for_status()
            return res.json()

        def _done(data: dict):
            if source == "robot-ollama" and structured:
                self._last_structured = data
                self.llm_output.setPlainText("")
                self.structured_output.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
                if self.llm_speak.isChecked() and data.get("text"):
                    self._speak_llm(data.get("text"))
                return
            self.llm_output.setPlainText(data.get("response", data.get("text", "")))
            if structured:
                self.structured_output.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            if self.llm_speak.isChecked():
                self._speak_llm(self.llm_output.toPlainText())

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        worker.signals.error.connect(lambda e: self.llm_output.setPlainText(f"Error: {e}"))
        self.thread_pool.start(worker)

    def _refresh_personas(self):
        def _do():
            res = requests.get(f"{self.robot_endpoints.gateway_base()}/ollama/personas", timeout=4)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(self._update_personas)
        worker.signals.error.connect(lambda e: self._log(f"Persona error: {e}"))
        self.thread_pool.start(worker)

    def _update_personas(self, payload: dict):
        items = payload.get("items", [])
        active = payload.get("active")
        self.persona_select.blockSignals(True)
        self.persona_select.clear()
        self.persona_select.addItems(items)
        if active and active in items:
            self.persona_select.setCurrentText(active)
        self.persona_select.blockSignals(False)

    def _select_persona(self, name: str):
        if not name:
            return

        def _do():
            res = requests.post(
                f"{self.robot_endpoints.gateway_base()}/ollama/persona/select",
                params={"name": name},
                timeout=4,
            )
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda _: self._log(f"Persona seçildi: {name}"))
        worker.signals.error.connect(lambda e: self._log(f"Persona error: {e}"))
        self.thread_pool.start(worker)

    def _stt_start(self):
        def _do():
            res = requests.post(f"{self.robot_endpoints.gateway_base()}/speech/start", timeout=3)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda _: self._log("STT başladı"))
        worker.signals.error.connect(lambda e: self._log(f"STT error: {e}"))
        self.thread_pool.start(worker)

        if not hasattr(self, "_stt_timer"):
            self._stt_timer = QtCore.QTimer(self)
            self._stt_timer.setInterval(1200)
            self._stt_timer.timeout.connect(self._stt_poll)
        self._stt_timer.start()

    def _stt_stop(self):
        def _do():
            res = requests.post(f"{self.robot_endpoints.gateway_base()}/speech/stop", timeout=3)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda _: self._log("STT durdu"))
        worker.signals.error.connect(lambda e: self._log(f"STT error: {e}"))
        self.thread_pool.start(worker)

        if hasattr(self, "_stt_timer"):
            self._stt_timer.stop()

    def _stt_poll(self):
        def _do():
            res = requests.get(f"{self.robot_endpoints.gateway_base()}/speech/last", timeout=2)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(self._stt_update)
        worker.signals.error.connect(lambda _: None)
        self.thread_pool.start(worker)

    def _stt_update(self, payload: dict):
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        self.stt_last.setText(f"STT: {text[:80]}")
        if not self.stt_auto_tts.isChecked():
            return
        lang = self._detect_lang(text)
        if self.tts_engine.currentText() == "piper":
            self.tts_lang.setCurrentText(lang)
            self._auto_select_piper_voice(lang)

    def _detect_lang(self, text: str) -> str:
        # basit heuristik
        if any(ch in text for ch in "çğıöşüÇĞİÖŞÜ"):
            return "tr"
        if re.search(r"\b(the|and|you|are|have|what|this|that)\b", text.lower()):
            return "en"
        return "tr"

    def _auto_select_piper_voice(self, lang: str):
        for i in range(self.piper_voice.count()):
            name = str(self.piper_voice.itemText(i)).lower()
            if f"{lang}_" in name or f"/{lang}/" in name or f"\\{lang}\\" in name:
                self.piper_voice.setCurrentIndex(i)
                break

    def _apply_structured_actions(self):
        if not self._last_structured:
            self.structured_output.setPlainText("Structured data yok")
            return

        payload = {
            "text": self._last_structured.get("text", ""),
            "actions": self._last_structured.get("actions"),
            "raw": self._last_structured.get("raw"),
            "speak": False,
        }

        def _do():
            res = requests.post(f"{self.robot_endpoints.gateway_base()}/autonomy/apply_actions", json=payload, timeout=10)
            res.raise_for_status()
            return res.json()

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda data: self.structured_output.append(f"\nUygulandı: {data}"))
        worker.signals.error.connect(lambda e: self.structured_output.append(f"\nApply error: {e}"))
        self.thread_pool.start(worker)

    def _speak_llm(self, text: str):
        if not text:
            return
        engine = self.tts_engine.currentText()
        voice = None
        language = self.tts_lang.currentText() or "tr"
        if engine == "piper":
            voice = self.piper_voice.currentData()
        else:
            voice = self.xtts_voice_path.text().strip() or None
        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "engine": engine,
        }

        def _do():
            return self._synthesize_local(text, engine, voice, language)

        def _done(data: bytes):
            if self.tts_play_on_robot.isChecked():
                self._play_on_robot(data)
            else:
                self._play_on_pc(data)

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        worker.signals.error.connect(lambda e: self._log(f"LLM speak error: {e}"))
        self.thread_pool.start(worker)

    def _synthesize_local(self, text: str, engine: str, voice: str | None, language: str) -> bytes:
        if engine == "piper":
            return self._synthesize_piper(text, voice)
        return self._synthesize_xtts(text, voice, language)

    def _synthesize_piper(self, text: str, model_path: str | None) -> bytes:
        path = model_path or self._piper_model_path
        if not path:
            raise RuntimeError("Piper model seçilmedi")

        try:
            from piper import PiperVoice
            if self._piper_model is None or self._piper_model_path != path:
                self._piper_model = PiperVoice.load(path, config_path=path + ".json", use_cuda=False)
                self._piper_model_path = path

            audio_chunks = self._piper_model.synthesize(text)
            pcm_data = b"".join(chunk.audio_int16_bytes for chunk in audio_chunks)
            import io, wave
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(pcm_data)
                return wav_buffer.getvalue()
        except Exception as exc:
            if "espeakbridge" not in str(exc).lower():
                raise

        base_dir = os.path.join(os.path.dirname(__file__), "tts", "TTS", "PiperTTS")
        exe_path = os.path.join(base_dir, "piper.exe")
        if not os.path.exists(exe_path):
            raise RuntimeError("Piper CLI bulunamadı")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        try:
            cmd = [exe_path, "--model", path, "--output_file", tmp_path]
            subprocess.run(
                cmd,
                input=text,
                text=True,
                check=True,
                cwd=base_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _synthesize_xtts(self, text: str, speaker_wav: str | None, language: str) -> bytes:
        self._ensure_xtts_loaded()

        if not speaker_wav or not os.path.exists(speaker_wav):
            raise RuntimeError("XTTS için speaker wav seçilmedi")

        import io
        import tempfile
        import numpy as np
        import soundfile as sf

        chunks = self._split_tts_text(text, settings.xtts_max_chars_per_chunk)
        audio_parts: list[np.ndarray] = []
        sample_rate = None

        for chunk in chunks:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self._xtts_model.tts_to_file(text=chunk, language=language, file_path=tmp_path, speaker_wav=speaker_wav)
                data, sr = sf.read(tmp_path, dtype="float32")
                if sample_rate is None:
                    sample_rate = sr
                audio_parts.append(data)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        if not audio_parts:
            raise RuntimeError("XTTS ses üretilemedi")

        audio = np.concatenate(audio_parts)
        with io.BytesIO() as buff:
            sf.write(buff, audio, sample_rate or 22050, format="WAV")
            return buff.getvalue()

    def _split_tts_text(self, text: str, max_chars: int) -> list[str]:
        cleaned = " ".join(text.split())
        if max_chars <= 0 or len(cleaned) <= max_chars:
            return [cleaned]

        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        chunks: list[str] = []
        current = ""
        for part in parts:
            if not part:
                continue
            if len(current) + len(part) + 1 <= max_chars:
                current = f"{current} {part}".strip()
                continue
            if current:
                chunks.append(current)
            if len(part) > max_chars:
                for i in range(0, len(part), max_chars):
                    chunks.append(part[i : i + max_chars])
                current = ""
            else:
                current = part
        if current:
            chunks.append(current)
        return chunks

    # ---------------- Terminal ----------------
    def _terminal_send(self):
        cmd = self.terminal_input.text().strip()
        if not cmd:
            return
        self.terminal_output.append(f"> {cmd}")
        self.terminal_input.clear()

        if cmd.lower() == "exit":
            self._close_ssh()
            return

        if cmd.lower().startswith("ssh "):
            self._handle_ssh_connect(cmd)
            return

        self._ssh_exec(cmd)

    def _handle_ssh_connect(self, cmd: str):
        match = re.search(r"ssh\s+([\w.-]+)@([\w.-]+)(?:\s+(\S+))?", cmd)
        if not match:
            self.terminal_output.append("Kullanım: ssh user@host")
            return
        user = match.group(1)
        host = match.group(2)
        password = match.group(3)
        def _do():
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            if password:
                client.connect(hostname=host, username=user, password=password, timeout=10)
            else:
                client.connect(hostname=host, username=user, timeout=10)
            return client

        def _done(client):
            with self._ssh_lock:
                self._ssh_client = client
            self.terminal_output.append("SSH bağlantısı hazır.")

        worker = HttpWorker(_do)
        worker.signals.done.connect(_done)
        worker.signals.error.connect(lambda e: self.terminal_output.append(f"SSH error: {e}"))
        self.thread_pool.start(worker)

    def _ssh_exec(self, cmd: str):
        def _do():
            with self._ssh_lock:
                client = self._ssh_client
            if client is None:
                raise RuntimeError("SSH bağlı değil. Önce ssh user@host")
            _, stdout, stderr = client.exec_command(cmd, timeout=20)
            out = stdout.read().decode("utf-8", errors="ignore")
            err = stderr.read().decode("utf-8", errors="ignore")
            return out + err

        worker = HttpWorker(_do)
        worker.signals.done.connect(lambda data: self.terminal_output.append(data.strip()))
        worker.signals.error.connect(lambda e: self.terminal_output.append(f"SSH error: {e}"))
        self.thread_pool.start(worker)

    def _close_ssh(self):
        with self._ssh_lock:
            if self._ssh_client:
                try:
                    self._ssh_client.close()
                except Exception:
                    pass
                self._ssh_client = None
        self.terminal_output.append("SSH bağlantısı kapandı.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._stop_local_camera()
        except Exception:
            pass
        try:
            self._close_ssh()
        except Exception:
            pass
        super().closeEvent(event)


__all__ = ["RobotControlWindow"]
