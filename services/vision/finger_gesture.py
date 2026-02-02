import logging
from typing import Dict, List, Optional

import cv2

logger = logging.getLogger(__name__)


class FingerGestureDetector:
    FINGERTIPS = [4, 8, 12, 16, 20]

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
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("Finger gesture detector loaded.")
        except Exception as exc:
            logger.warning("Finger gesture load failed: %s", exc)
            self.hands = None

    def detect(self, img) -> Dict[str, Optional[str]]:
        if not self.hands:
            return {"command": None, "hands": []}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        hands_list: List[Dict] = []
        hand_types: List[str] = []
        hand_states: List[str] = []

        if res.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(res.multi_hand_landmarks[:2]):
                label = "Unknown"
                if res.multi_handedness:
                    label = res.multi_handedness[i].classification[0].label
                hand_types.append(label)

                lm = hand_landmarks.landmark
                states = [0] * 5
                # Thumb
                if label == "Right":
                    states[0] = 1 if lm[self.FINGERTIPS[0]].x > lm[self.FINGERTIPS[0] - 1].x else 0
                else:
                    states[0] = 1 if lm[self.FINGERTIPS[0]].x < lm[self.FINGERTIPS[0] - 1].x else 0
                # Other fingers
                for f in range(1, 5):
                    states[f] = 1 if lm[self.FINGERTIPS[f]].y < lm[self.FINGERTIPS[f] - 2].y else 0
                hand_states.append("".join(map(str, states)))

                landmarks = [
                    {"x": float(pt.x), "y": float(pt.y), "z": float(pt.z)}
                    for pt in lm
                ]

                hands_list.append({
                    "type": label,
                    "state": hand_states[-1],
                    "landmarks": landmarks,
                })

        command = self._determine_command(hand_types, hand_states)
        return {"command": command, "hands": hands_list}

    def _determine_command(self, hand_types: List[str], hand_states: List[str]) -> Optional[str]:
        if len(hand_states) == 1:
            return f"{hand_types[0]}:{hand_states[0]}"
        if len(hand_states) == 2:
            try:
                l_idx = hand_types.index("Left")
                r_idx = hand_types.index("Right")
                return f"Left:{hand_states[l_idx]}_Right:{hand_states[r_idx]}"
            except ValueError:
                return None
        return None
