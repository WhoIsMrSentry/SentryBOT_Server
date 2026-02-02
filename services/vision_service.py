import logging
import cv2
import numpy as np
import requests
from typing import Dict, Any, List, Optional
from services.base import BaseService
from config import settings
from services.vision import (
    ObjectDetector,
    HandDetector,
    FaceAnalyzer,
    AgeEmotionDetector,
    MotionDetector,
    FingerGestureDetector,
    FaceRecognizer,
)

logger = logging.getLogger(__name__)


class VisionService(BaseService):
    def __init__(self):
        self.object_detector = ObjectDetector(settings.yolo_model_path)
        self.hand_detector = HandDetector()
        self.face_analyzer = FaceAnalyzer()
        self.age_emotion_detector = AgeEmotionDetector()
        self.motion_detector = MotionDetector()
        self.finger_gesture_detector = FingerGestureDetector()
        self.face_recognizer = FaceRecognizer()

    def initialize(self):
        if not settings.vision_enabled:
            return
        self.object_detector.load()
        self.hand_detector.load()
        self.face_analyzer.load()
        self.age_emotion_detector.load()
        self.motion_detector.load()
        self.finger_gesture_detector.load()
        self.face_recognizer.load()

    def process_image(self, image_bytes: bytes, mode: Optional[str] = None, modalities: List[str] = None) -> Dict[str, Any]:
        """
        Process image bytes.
        mode: single 'object' | 'face' | 'hand' | 'attributes'
        modalities: list (deprecated) -> only one allowed
        """
        if mode:
            modes = [mode]
        else:
            modes = modalities or ["object"]
        modes = [m for m in modes if m]
        if len(modes) != 1:
            raise ValueError("Only one processing mode is allowed")
        mode = modes[0]

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        results = {}

        if mode == "object":
            if not self.object_detector.available:
                results["error"] = "object detector not available"
            else:
                results["objects"] = self.object_detector.detect(img)
        elif mode == "hand":
            if not self.hand_detector.available:
                results["error"] = "hand detector not available"
            else:
                results["hands"] = self.hand_detector.detect(img)
        elif mode == "face":
            if not self.face_analyzer.available:
                results["error"] = "face analyzer not available"
            else:
                results["faces"] = self.face_analyzer.analyze(img, "face")
        elif mode == "attributes":
            if not self.face_analyzer.available:
                results["error"] = "face analyzer not available"
            else:
                results["faces"] = self.face_analyzer.analyze(img, "attributes")
        elif mode == "age_emotion":
            if not self.age_emotion_detector.available:
                results["error"] = "age/emotion detector not available"
            else:
                results["faces"] = self.age_emotion_detector.process(img)
        elif mode == "motion":
            if not self.motion_detector.processing_active:
                self.motion_detector.start()
            results["motion"] = self.motion_detector.process(img)
        elif mode == "finger":
            if not self.finger_gesture_detector.available:
                results["error"] = "finger gesture detector not available"
            else:
                results["gesture"] = self.finger_gesture_detector.detect(img)
        elif mode == "face_recognize":
            if not self.face_recognizer.available:
                results["error"] = "face recognizer not available"
            else:
                results["faces"] = self.face_recognizer.recognize(img)
        else:
            results["error"] = f"unknown mode: {mode}"

        self._push_results(mode, results)
        return results

    def _push_results(self, mode: str | None, results: Dict[str, Any]) -> None:
        if not settings.robot_gateway_url:
            return
        objects = results.get("objects")
        if not objects:
            return
        url = f"{settings.robot_gateway_url}/vision/results"
        payload = {"objects": objects, "mode": mode}
        headers = {}
        if settings.robot_vision_auth_token:
            headers["X-Auth-Token"] = settings.robot_vision_auth_token
        try:
            requests.post(url, json=payload, headers=headers, timeout=1.5)
        except Exception:
            pass

    def health_check(self) -> dict:
        return {
            "yolo": self.object_detector.available,
            "hands": self.hand_detector.available,
            "deepface": self.face_analyzer.available,
            "age_emotion": self.age_emotion_detector.available,
            "motion": True,
            "finger": self.finger_gesture_detector.available,
            "face_recognize": self.face_recognizer.available,
        }
