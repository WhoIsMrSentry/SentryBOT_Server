import logging
from typing import Any, Dict, List
import cv2

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    def __init__(self):
        self.DeepFace = None

    @property
    def available(self) -> bool:
        return self.DeepFace is not None

    def load(self) -> None:
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            logger.info("DeepFace module loaded.")
        except Exception as e:
            logger.warning(f"DeepFace load failed: {e}")
            self.DeepFace = None

    def analyze(self, img, mode: str) -> List[Dict[str, Any]]:
        if not self.DeepFace:
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            if mode == "attributes":
                actions = ["age", "gender", "emotion"]
                analysis = self.DeepFace.analyze(
                    img_path=img_rgb,
                    actions=actions,
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(analysis, list):
                    return analysis
                return [analysis]
            # face only
            faces = self.DeepFace.extract_faces(img_path=img_rgb, enforce_detection=False)
            out = []
            for f in faces:
                area = f.get("facial_area") or {}
                out.append({
                    "facial_area": area,
                    "confidence": f.get("confidence"),
                })
            return out
        except Exception as e:
            logger.error(f"DeepFace processing error: {e}")
            return [{"error": str(e)}]
