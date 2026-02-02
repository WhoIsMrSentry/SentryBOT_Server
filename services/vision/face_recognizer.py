import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(self):
        self.cascade = None
        self.recognizer = None
        self.encodings: Optional[dict] = None
        self.face_recognition = None
        self.use_encodings = False
        self.labels: Dict[int, str] = {}
        self.available = False

    def load(self) -> None:
        try:
            enc_path = settings.vision_face_encodings_path
            if os.path.exists(enc_path):
                try:
                    import face_recognition  # type: ignore
                    with open(enc_path, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and "encodings" in data and "names" in data:
                        self.encodings = data
                        self.face_recognition = face_recognition
                        self.use_encodings = True
                        self.available = True
                        logger.info("Face encodings loaded: %d", len(data.get("names", [])))
                        return
                except Exception as exc:
                    logger.warning("Encodings load failed, fallback to cascade: %s", exc)

            if os.path.exists(settings.vision_face_cascade_path):
                self.cascade = cv2.CascadeClassifier(settings.vision_face_cascade_path)
            else:
                logger.warning("Cascade not found: %s", settings.vision_face_cascade_path)
                self.cascade = None

            if self.cascade is None or self.cascade.empty():
                logger.warning("Haar cascade failed to load.")
                self.cascade = None

            if self.cascade is None:
                self.available = False
                return

            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces, labels, label_map = self._load_dataset(settings.vision_face_dataset_dir)
            if faces:
                self.recognizer.train(faces, np.array(labels))
                self.labels = label_map
                self.available = True
                logger.info("Face recognizer trained with %d samples.", len(faces))
            else:
                self.available = False
                logger.warning("No face dataset found for recognition.")
        except Exception as exc:
            logger.warning("Face recognizer load failed: %s", exc)
            self.available = False

    def recognize(self, img) -> List[Dict]:
        if not self.available:
            return []

        if self.use_encodings and self.encodings and self.face_recognition:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = self.face_recognition.face_locations(rgb, model="hog")
            encs = self.face_recognition.face_encodings(rgb, boxes)
            results: List[Dict] = []
            known_enc = self.encodings.get("encodings", [])
            known_names = self.encodings.get("names", [])
            for (top, right, bottom, left), enc in zip(boxes, encs):
                name = "unknown"
                confidence = 1.0
                if known_enc:
                    matches = self.face_recognition.compare_faces(
                        known_enc,
                        enc,
                        tolerance=settings.vision_face_encoding_tolerance,
                    )
                    distances = self.face_recognition.face_distance(known_enc, enc)
                    if len(distances) > 0:
                        best_idx = int(np.argmin(distances))
                        confidence = float(distances[best_idx])
                        if matches[best_idx]:
                            name = known_names[best_idx]
                is_owner = bool(name != "unknown" and name.lower() == settings.vision_owner_name.lower())
                results.append({
                    "name": name,
                    "is_owner": is_owner,
                    "confidence": float(confidence),
                    "bbox": [int(left), int(top), int(right - left), int(bottom - top)],
                })
            return results

        if self.cascade is None or self.recognizer is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        results: List[Dict] = []
        for (x, y, w, h) in faces:
            face_img = gray[y : y + h, x : x + w]
            face_img = cv2.resize(face_img, (200, 200))
            label_id, confidence = self.recognizer.predict(face_img)
            name = self.labels.get(label_id, "unknown")
            if confidence > settings.vision_face_recognize_threshold:
                name = "unknown"
            is_owner = bool(name != "unknown" and name.lower() == settings.vision_owner_name.lower())
            results.append({
                "name": name,
                "is_owner": is_owner,
                "confidence": float(confidence),
                "bbox": [int(x), int(y), int(w), int(h)],
            })
        return results

    def _load_dataset(self, base_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
        faces: List[np.ndarray] = []
        labels: List[int] = []
        label_map: Dict[int, str] = {}
        if not os.path.isdir(base_dir):
            return faces, labels, label_map

        label_id = 0
        for name in sorted(os.listdir(base_dir)):
            person_dir = os.path.join(base_dir, name)
            if not os.path.isdir(person_dir):
                continue
            label_map[label_id] = name
            for file in os.listdir(person_dir):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(label_id)
            label_id += 1
        return faces, labels, label_map
