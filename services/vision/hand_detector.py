import logging
from typing import Dict, List
import cv2

logger = logging.getLogger(__name__)


class HandDetector:
    def __init__(self):
        self.hands = None
        self.mp_hands = None

    @property
    def available(self) -> bool:
        return self.hands is not None

    def load(self) -> None:
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                self.mp_hands = mp.solutions.hands
            else:
                from mediapipe.python.solutions import hands as mp_hands
                self.mp_hands = mp_hands
            self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            logger.info("MediaPipe Hands loaded.")
        except Exception as e:
            logger.warning(f"MediaPipe Hands load failed: {e}")
            self.hands = None

    def detect(self, img) -> List[Dict]:
        if not self.hands:
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        hands_list: List[Dict] = []
        if res.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
                label = "Unknown"
                score = 0.0
                if res.multi_handedness:
                    label = res.multi_handedness[i].classification[0].label
                    score = res.multi_handedness[i].classification[0].score
                landmarks = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in hand_landmarks.landmark
                ]
                hands_list.append({
                    "label": label,
                    "confidence": score,
                    "landmarks_count": len(hand_landmarks.landmark),
                    "landmarks": landmarks,
                })
        return hands_list
