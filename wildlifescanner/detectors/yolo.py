from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import logging
import numpy as np
import torch
from ultralytics import YOLO

from ..models import Detection
from .base import AnimalDetector


logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Select best available device: prefer MPS (Apple) > CUDA > CPU."""
    try:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        # Any probing issue -> fall back to CPU
        pass
    return "cpu"


class YOLODetector(AnimalDetector):
    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        allowed_classes: Tuple[str, ...] | None = None,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.device = _select_device()
        logger.info("YOLODetector using device: %s", self.device)
        # Map class id -> name
        self.class_names = {int(k): v for k, v in self.model.model.names.items()} if hasattr(self.model, "model") else self.model.names  # type: ignore[attr-defined]
        if isinstance(self.class_names, list):
            self.class_names = {i: name for i, name in enumerate(self.class_names)}  # type: ignore[assignment]
        self.allowed = set(c.lower() for c in allowed_classes) if allowed_classes else None

    def warmup(self) -> None:  # pragma: no cover (fast path, optional)
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = self.model.predict(
            dummy,
            imgsz=320,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )

    def detect(self, frame_bgr: np.ndarray) -> Sequence[Detection]:
        # Ultralytics expects RGB
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.model.predict(
            frame_rgb,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else None
        if xyxy is None or conf is None or cls is None:
            return out
        for (x1, y1, x2, y2), sc, ci in zip(xyxy, conf, cls):
            name = str(self.class_names.get(int(ci), str(int(ci)))).lower()
            if self.allowed is not None and name not in self.allowed:
                continue
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(sc), name))
        return out

    def close(self) -> None:  # pragma: no cover
        # Nothing to close for Ultralytics
        return
