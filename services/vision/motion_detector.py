import logging
from typing import Dict, List, Tuple

import cv2

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self):
        self.processing_active = False
        self.min_area = 1200
        self.min_area_fallback = 800
        self.min_area_ratio = 0.002
        self.scale_factor = 0.5
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=80, detectShadows=True
        )
        self._last_gray = None

    @property
    def available(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def start(self) -> None:
        self.processing_active = True
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    def stop(self) -> None:
        self.processing_active = False

    def process(self, frame) -> Dict[str, List[Tuple[int, int, int, int]]]:
        if not self.processing_active or frame is None:
            return {"detected": False, "areas": []}

        motion_areas: List[Tuple[int, int, int, int]] = []
        frame_h, frame_w = frame.shape[:2]
        area_threshold = max(self.min_area, int(frame_h * frame_w * self.min_area_ratio))
        try:
            small = cv2.resize(frame, (int(frame_w * self.scale_factor), int(frame_h * self.scale_factor)))
            small = cv2.GaussianBlur(small, (7, 7), 0)
            fg_mask = self.background_subtractor.apply(small)
            fg_mask[fg_mask == 127] = 0
            thresh_mask = cv2.threshold(fg_mask, 35, 255, cv2.THRESH_BINARY)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morph_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            morph_mask = cv2.dilate(morph_mask, kernel, iterations=1)

            contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < area_threshold * (self.scale_factor ** 2):
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                # scale back to original size
                x = int(x / self.scale_factor)
                y = int(y / self.scale_factor)
                w = int(w / self.scale_factor)
                h = int(h / self.scale_factor)
                motion_areas.append((x, y, w, h))
        except Exception as exc:
            logger.warning("Motion detection error: %s", exc)
            return {"detected": False, "areas": []}

        if not motion_areas:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            if self._last_gray is not None:
                diff = cv2.absdiff(self._last_gray, gray)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < max(self.min_area_fallback, area_threshold):
                        continue
                    (x, y, w, h) = cv2.boundingRect(contour)
                    motion_areas.append((x, y, w, h))
            self._last_gray = gray

        return {"detected": bool(motion_areas), "areas": motion_areas}
