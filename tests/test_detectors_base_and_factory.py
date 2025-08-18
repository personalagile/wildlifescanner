from __future__ import annotations

import sys
import types

import numpy as np

from wildlifescanner.detectors.base import AnimalDetector
from wildlifescanner.detectors.factory import create_detector


class _Dummy(AnimalDetector):
    def warmup(self) -> None:  # pragma: no cover - unused in test
        pass

    def detect(self, frame_bgr: np.ndarray):
        # Return coordinates depending on sum to make deterministic
        s = int(frame_bgr.sum())
        return [types.SimpleNamespace(x1=s, y1=s, x2=s + 1, y2=s + 1, score=0.9, cls_name="bird")]

    def close(self) -> None:  # pragma: no cover - unused in test
        pass


def test_detect_many_yields_sequences():
    det = _Dummy()
    frames = [np.zeros((1, 1, 3), dtype=np.uint8), np.ones((1, 1, 3), dtype=np.uint8)]
    outs = list(det.detect_many(frames))
    assert len(outs) == 2
    assert len(outs[0]) == 1 and len(outs[1]) == 1


def test_factory_yolo_branch_sysmodules_stub(monkeypatch):
    # Stub YOLODetector in sys.modules to avoid importing ultralytics
    mod = types.SimpleNamespace()

    class _YOLO:
        def __init__(self, **kwargs):
            self.kw = kwargs

        def close(self):
            pass

    mod.YOLODetector = _YOLO
    sys.modules["wildlifescanner.detectors.yolo"] = mod

    det = create_detector(
        "yolo",
        yolo_model="model.pt",
        confidence=0.3,
        iou=0.5,
        allowed_classes=("bird",),
    )

    assert isinstance(det, _YOLO)
    assert det.kw["model_path"] == "model.pt"
    assert det.kw["confidence"] == 0.3
    assert det.kw["iou"] == 0.5
    assert det.kw["allowed_classes"] == ("bird",)
