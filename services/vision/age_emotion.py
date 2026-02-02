import logging
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class AgeEmotionDetector:
    def __init__(self):
        self.face_detector = None
        self.age_net = None
        self.emotion_net = None
        self.available = False

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
        self.emotion_list = ["Notr", "Mutlu", "Saskin", "Uzgun", "Kizgin", "Igrenmis", "Korkmus", "Kucumseme"]

        self.padding = 20
        self.last_detection_time = 0.0
        self.detection_interval = 0.5

        self.last_results: List[Dict[str, Any]] = []
        self.last_results_time = 0.0
        self.max_persist_time = 1.5

    def load(self) -> None:
        try:
            face_model = settings.vision_face_model_path
            if os.path.exists(face_model):
                self.face_detector = cv2.FaceDetectorYN_create(face_model, "", (0, 0))
                logger.info("Face detector loaded.")
            else:
                logger.warning("Face model not found: %s", face_model)

            if os.path.exists(settings.vision_age_model_path) and os.path.exists(settings.vision_age_proto_path):
                self.age_net = cv2.dnn.readNet(settings.vision_age_model_path, settings.vision_age_proto_path)
                logger.info("Age model loaded.")
            else:
                logger.warning("Age model files missing.")

            if os.path.exists(settings.vision_emotion_model_path):
                self.emotion_net = cv2.dnn.readNet(settings.vision_emotion_model_path)
                logger.info("Emotion model loaded.")
            else:
                logger.warning("Emotion model not found: %s", settings.vision_emotion_model_path)

            self.available = bool(self.face_detector and self.age_net and self.emotion_net)
        except Exception as exc:
            logger.warning("Age/Emotion load failed: %s", exc)
            self.available = False

    def process(self, frame) -> List[Dict[str, Any]]:
        if frame is None:
            return []

        now = time.time()
        if not self.available:
            return self._persist_last(frame, now)[1]

        if now - self.last_detection_time < self.detection_interval:
            return self._persist_last(frame, now)[1]

        self.last_detection_time = now
        frame_h, frame_w = frame.shape[:2]
        results: List[Dict[str, Any]] = []

        try:
            self.face_detector.setInputSize((frame_w, frame_h))
            faces = self.face_detector.detect(frame)
            if faces[1] is None:
                return self._persist_last(frame, now)[1]

            for face_data in faces[1]:
                confidence = float(face_data[-1])
                if confidence < 0.7:
                    continue
                x, y, w, h = face_data[0:4].astype(np.int32)
                x1 = max(0, x - self.padding)
                y1 = max(0, y - self.padding)
                x2 = min(frame_w - 1, x + w + self.padding)
                y2 = min(frame_h - 1, y + h + self.padding)
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                age = "Unknown"
                emotion = "Unknown"

                if self.age_net is not None:
                    age_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                    self.age_net.setInput(age_blob)
                    age_preds = self.age_net.forward()
                    age = self.age_list[int(age_preds[0].argmax())]

                if self.emotion_net is not None:
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    emotion_roi = cv2.resize(face_gray, (64, 64))
                    emotion_blob = cv2.dnn.blobFromImage(emotion_roi, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
                    self.emotion_net.setInput(emotion_blob)
                    emotion_preds = self.emotion_net.forward()
                    emotion = self.emotion_list[int(np.argmax(emotion_preds[0]))]

                results.append({
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "age": age,
                    "emotion": emotion,
                    "confidence": confidence,
                })

            if results:
                self.last_results = results
                self.last_results_time = now
            return results
        except Exception as exc:
            logger.warning("Age/Emotion processing error: %s", exc)
            return self._persist_last(frame, now)[1]

    def _persist_last(self, frame, now: float) -> Tuple[Any, List[Dict[str, Any]]]:
        if self.last_results and (now - self.last_results_time < self.max_persist_time):
            return frame, self.last_results
        self.last_results = []
        return frame, []
