from __future__ import annotations

from .base import AnimalDetector


def create_detector(
    name: str,
    *,
    yolo_model: str,
    confidence: float,
    iou: float,
    allowed_classes: tuple[str, ...] | None,
) -> AnimalDetector:
    name_u = name.upper()
    if name_u == "YOLO":
        # Lazy import so tests that don't use YOLO won't require ultralytics
        from .yolo import YOLODetector

        return YOLODetector(
            model_path=yolo_model,
            confidence=confidence,
            iou=iou,
            allowed_classes=allowed_classes,
        )
    elif name_u == "MEGADETECTOR":
        # Lazy import so tests can stub this module without ultralytics installed
        from .megadetector import MegaDetector

        return MegaDetector(
            model_path=yolo_model,
            confidence=confidence,
            iou=iou,
            allowed_classes=allowed_classes,
        )
    else:
        raise ValueError(f"Unknown detector: {name}")
