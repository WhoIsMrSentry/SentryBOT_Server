import logging
import cv2
import numpy as np
from typing import Dict, Any, List
from services.base import BaseService
from config import settings

logger = logging.getLogger(__name__)

class VisionService(BaseService):
    def __init__(self):
        self.yolo_model = None
        self.face_model = None # DeepFace or similar
        self.hands = None # MediaPipe

    def initialize(self):
        if not settings.vision_enabled:
            return

        # 1. YOLO Object Detection
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(settings.yolo_model_path)
            logger.info("YOLO model loaded.")
        except Exception as e:
            logger.warning(f"YOLO load failed: {e}")

        # 2. MediaPipe Hands
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            logger.info("MediaPipe Hands loaded.")
        except Exception as e:
            logger.warning(f"MediaPipe Hands load failed: {e}")
        
        # 3. DeepFace (Face Rec + Attributes)
        # DeepFace loads models on first use usually, but we can try to pre-import
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            logger.info("DeepFace module loaded.")
        except Exception as e:
            logger.warning(f"DeepFace load failed: {e}")
            self.DeepFace = None

    def process_image(self, image_bytes: bytes, modalities: List[str] = None) -> Dict[str, Any]:
        """
        Process image bytes.
        modalities: list of 'object', 'face', 'hand', 'attributes'
        """
        if modalities is None:
            modalities = ["object", "face", "hand", "attributes"]

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        results = {}

        # OBJECTS
        if "object" in modalities and self.yolo_model:
            results["objects"] = self._detect_objects(img)

        # HANDS
        if "hand" in modalities and self.hands:
            results["hands"] = self._detect_hands(img)

        # FACE / ATTRIBUTES
        if ("face" in modalities or "attributes" in modalities) and self.DeepFace:
            # DeepFace expects path or numpy array (BGR is fine for opencv backend usually, but DeepFace might want RGB)
            # DeepFace.analyze accepts numpy array (BGR by default if enforce_detection=False?)
            # Let's convert to RGB for safety.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                # actions: ['age', 'gender', 'race', 'emotion']
                actions = []
                if "attributes" in modalities:
                    actions.extend(['age', 'gender', 'emotion'])
                
                # If we just want face detection/recognition, we might use represent or extract_faces
                # But analyze gives us bounding box + attributes.
                if actions:
                    analysis = self.DeepFace.analyze(img_path=img_rgb, actions=actions, enforce_detection=False, silent=True)
                    if isinstance(analysis, list):
                        results["faces"] = analysis
                    else:
                        results["faces"] = [analysis]
            except Exception as e:
                logger.error(f"DeepFace processing error: {e}")
                results["faces_error"] = str(e)

        return results

    def _detect_objects(self, img) -> List[Dict]:
        res = self.yolo_model(img, verbose=False)
        objects = []
        for r in res:
            for box in r.boxes:
                try:
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    raw_cls = int(box.cls[0])
                    label = self.yolo_model.names.get(raw_cls, str(raw_cls))
                    objects.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": coords
                    })
                except: pass
        return objects

    def _detect_hands(self, img) -> List[Dict]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        hands_list = []
        if res.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
                # Basic info: which hand (Label), confidence
                label = "Unknown"
                score = 0.0
                if res.multi_handedness:
                    label = res.multi_handedness[i].classification[0].label
                    score = res.multi_handedness[i].classification[0].score
                
                # Check for simple gestures (e.g. open palm vs fist) - sophisticated gesture logic to be added
                # For now just return existence
                hands_list.append({
                    "label": label,
                    "confidence": score,
                    "landmarks_count": len(hand_landmarks.landmark)
                })
        return hands_list

    def health_check(self) -> dict:
        return {
            "yolo": self.yolo_model is not None,
            "hands": self.hands is not None,
            "deepface": self.DeepFace is not None
        }
