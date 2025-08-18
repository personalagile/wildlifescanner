from __future__ import annotations

import pytest

from wildlifescanner.detectors.factory import create_detector


def test_factory_megadetector_not_implemented():
    with pytest.raises(NotImplementedError):
        create_detector(
            "MEGADETECTOR",
            yolo_model="yolov8n.pt",
            confidence=0.25,
            iou=0.45,
            allowed_classes=None,
        )


def test_factory_invalid_detector_raises_value_error():
    with pytest.raises(ValueError):
        create_detector(
            "UNKNOWN",
            yolo_model="yolov8n.pt",
            confidence=0.25,
            iou=0.45,
            allowed_classes=None,
        )
