from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO  # type: ignore[import-untyped]

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
        pass
    return "cpu"


class MegaDetector(AnimalDetector):
    """
    PyTorch MegaDetector wrapper.

    Notes:
    - Loads a YOLO-style .pt model with class names typically {animal, person, vehicle}.
    - Works with MDv5 weights and, for testing, can run with any compatible Ultralytics model path.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        allowed_classes: tuple[str, ...] | None = None,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.device = _select_device()
        logger.info("MegaDetector using device: %s", self.device)
        logger.info("MegaDetector model: %s", model_path)
        # Class mapping
        if hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            names = self.model.model.names  # type: ignore[attr-defined]
        else:
            names = getattr(self.model, "names", {})
        if isinstance(names, list):
            self.class_names = {i: name for i, name in enumerate(names)}
        else:
            self.class_names = {int(k): v for k, v in names.items()}
        # Log model classes and validate allowed classes against model
        model_class_set = {str(v).lower() for v in self.class_names.values()}
        logger.info("Model classes: %s", sorted(model_class_set))

        self.allowed = set(c.lower() for c in allowed_classes) if allowed_classes else None
        if self.allowed is None:
            logger.info("No class filter applied (allowed_classes=None)")
        else:
            unknown = sorted([c for c in self.allowed if c not in model_class_set])
            if unknown:
                logger.warning(
                    "Allowed classes not present in model: %s | model classes=%s",
                    unknown,
                    sorted(model_class_set),
                )
            logger.info("Filtering to classes: %s", sorted(self.allowed))

    def warmup(self) -> None:  # pragma: no cover (optional)
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
        raw_count = int(len(cls)) if hasattr(cls, "__len__") else 0
        for (x1, y1, x2, y2), sc, ci in zip(xyxy, conf, cls, strict=False):
            name = str(self.class_names.get(int(ci), str(int(ci)))).lower()
            if self.allowed is not None and name not in self.allowed:
                continue
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(sc), name))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Frame detections: raw=%d filtered=%d (allowed=%s)",
                raw_count,
                len(out),
                sorted(self.allowed) if self.allowed is not None else None,
            )
        return out

    def close(self) -> None:  # pragma: no cover
        return
