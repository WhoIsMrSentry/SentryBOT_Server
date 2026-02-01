import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        try:
            import torch
            from ultralytics import YOLO
            try:
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
            except Exception:
                pass
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded.")
        except Exception as e:
            logger.warning(f"YOLO load failed: {e}")
            self.model = None

    def detect(self, img) -> List[Dict]:
        if not self.model:
            return []
        res = self.model(img, verbose=False)
        objects: List[Dict] = []
        for r in res:
            for box in r.boxes:
                try:
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    raw_cls = int(box.cls[0])
                    label = self.model.names.get(raw_cls, str(raw_cls))
                    objects.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": coords,
                    })
                except Exception:
                    pass
        return objects
