from __future__ import annotations

import sys
import types

import pytest

from wildlifescanner.detectors.factory import create_detector


def test_factory_megadetector_branch_sysmodules_stub(monkeypatch):
    # Stub MegaDetector in sys.modules to avoid importing ultralytics
    mod = types.SimpleNamespace()

    class _MD:
        def __init__(self, **kwargs):
            self.kw = kwargs

        def close(self):
            pass

    mod.MegaDetector = _MD
    sys.modules["wildlifescanner.detectors.megadetector"] = mod

    det = create_detector(
        "MEGADETECTOR",
        yolo_model="md.pt",
        confidence=0.3,
        iou=0.5,
        allowed_classes=("animal",),
    )

    assert isinstance(det, _MD)
    assert det.kw["model_path"] == "md.pt"
    assert det.kw["confidence"] == 0.3
    assert det.kw["iou"] == 0.5
    assert det.kw["allowed_classes"] == ("animal",)


def test_factory_invalid_detector_raises_value_error():
    with pytest.raises(ValueError):
        create_detector(
            "UNKNOWN",
            yolo_model="yolov8n.pt",
            confidence=0.25,
            iou=0.45,
            allowed_classes=None,
        )
