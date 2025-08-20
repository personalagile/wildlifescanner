from __future__ import annotations

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def _load_megadetector_module(tmp_name: str = "_md_test"):
    """Load the MegaDetector module from source with stubbed heavy deps.

    This avoids interference from tests that stub the original module in sys.modules
    (e.g., tests/test_factory.py) and avoids importing real ultralytics/torch.
    """
    # Prepare minimal stubs so top-level imports succeed
    if "ultralytics" not in sys.modules:
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=object)  # placeholder
    if "torch" not in sys.modules:
        # Minimal structure for torch with mps/cuda checks
        sys.modules["torch"] = types.SimpleNamespace(
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
            cuda=types.SimpleNamespace(is_available=lambda: False),
        )

    md_path = (
        Path(__file__).resolve().parents[1] / "wildlifescanner" / "detectors" / "megadetector.py"
    )
    # Load the module under the real package so relative imports ("..models") work
    full_name = f"wildlifescanner.detectors.{tmp_name}"
    spec = spec_from_file_location(full_name, md_path)
    assert spec and spec.loader, "Failed to prepare spec for megadetector.py"
    mod = module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


class ArrayWrap:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):  # pragma: no cover - trivial
        return self

    def numpy(self):  # pragma: no cover - trivial
        return self._arr


class FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls_: np.ndarray):
        self.xyxy = ArrayWrap(xyxy)
        self.conf = ArrayWrap(conf)
        self.cls = ArrayWrap(cls_)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None):
        self.boxes = boxes


def test_select_device_prefers_mps_then_cuda_then_cpu(monkeypatch: pytest.MonkeyPatch):
    md = _load_megadetector_module("wd_md_select")

    class TorchStub:
        def __init__(self, mps: bool, cuda: bool):
            self.backends = types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: mps)
            )
            self.cuda = types.SimpleNamespace(is_available=lambda: cuda)

    # mps
    monkeypatch.setattr(md, "torch", TorchStub(mps=True, cuda=True))
    assert md._select_device() == "mps"
    # cuda
    monkeypatch.setattr(md, "torch", TorchStub(mps=False, cuda=True))
    assert md._select_device() == "cuda"
    # cpu
    monkeypatch.setattr(md, "torch", TorchStub(mps=False, cuda=False))
    assert md._select_device() == "cpu"


def test_detect_filters_and_converts_bgr_to_rgb(monkeypatch: pytest.MonkeyPatch):
    md = _load_megadetector_module("wd_md_detect1")

    # Stub device selection to be deterministic
    monkeypatch.setattr(md, "_select_device", lambda: "cpu")

    # Prepare a FakeYOLO with model.names list
    class FakeYOLO:
        def __init__(self, _path: str):  # pragma: no cover - trivial
            self.model = types.SimpleNamespace(names=["animal", "person"])
            self.last_img = None
            self.last_kwargs = None

        def predict(self, img, **kwargs):
            # Capture RGB input and params
            self.last_img = img
            self.last_kwargs = kwargs
            # Two boxes: class 0 (animal) and class 1 (person)
            xyxy = np.array([[10, 11, 20, 21], [30, 31, 40, 41]], dtype=float)
            conf = np.array([0.9, 0.8], dtype=float)
            cls_ = np.array([0, 1], dtype=float)
            return [FakeResult(FakeBoxes(xyxy, conf, cls_))]

    monkeypatch.setattr(md, "YOLO", FakeYOLO)

    det = md.MegaDetector("/tmp/dummy.pt", confidence=0.33, iou=0.44, allowed_classes=("animal",))

    # BGR frame with a simple identifiable pattern
    bgr = np.array([[[0, 1, 2]]], dtype=np.uint8)
    out = det.detect(bgr)

    # Ensure RGB conversion happened (reverse channels)
    assert np.array_equal(det.model.last_img, bgr[:, :, ::-1])
    # Ensure params were forwarded
    assert det.model.last_kwargs["conf"] == pytest.approx(0.33)
    assert det.model.last_kwargs["iou"] == pytest.approx(0.44)
    assert det.model.last_kwargs["verbose"] is False
    assert det.model.last_kwargs["device"] == "cpu"

    # Only the 'animal' detection should remain after filtering
    assert len(out) == 1
    d = out[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (10.0, 11.0, 20.0, 21.0)
    assert d.score == pytest.approx(0.9)
    assert d.cls_name == "animal"


def test_detect_with_names_dict_and_no_filter(monkeypatch: pytest.MonkeyPatch):
    md = _load_megadetector_module("wd_md_detect2")
    monkeypatch.setattr(md, "_select_device", lambda: "cpu")

    # FakeYOLO with top-level names dict (no .model attribute)
    class FakeYOLO:
        def __init__(self, _path: str):  # pragma: no cover - trivial
            self.names = {"0": "animal", "1": "vehicle"}

        def predict(self, img, **kwargs):  # pragma: no cover - minimal
            xyxy = np.array([[1, 2, 3, 4]], dtype=float)
            conf = np.array([0.5], dtype=float)
            cls_ = np.array([1], dtype=float)  # -> "vehicle"
            return [FakeResult(FakeBoxes(xyxy, conf, cls_))]

    monkeypatch.setattr(md, "YOLO", FakeYOLO)

    det = md.MegaDetector("/tmp/dummy.pt", confidence=0.2, iou=0.3)
    out = det.detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].cls_name == "vehicle"


def test_warmup_calls_predict(monkeypatch: pytest.MonkeyPatch):
    md = _load_megadetector_module("wd_md_warmup")
    monkeypatch.setattr(md, "_select_device", lambda: "cpu")

    class FakeYOLO:
        def __init__(self, _path: str):  # pragma: no cover - trivial
            self.calls = []

        def predict(self, img, **kwargs):
            # Record image shape and key params
            self.calls.append(
                (
                    img.shape,
                    kwargs.get("imgsz"),
                    kwargs.get("conf"),
                    kwargs.get("iou"),
                )
            )
            return [FakeResult(None)]

    monkeypatch.setattr(md, "YOLO", FakeYOLO)

    det = md.MegaDetector("/tmp/dummy.pt", confidence=0.4, iou=0.6)
    det.warmup()
    # One call with imgsz=320
    assert len(det.model.calls) == 1
    shape, imgsz, conf, iou = det.model.calls[0]
    assert shape == (320, 320, 3)
    assert imgsz == 320
    assert conf == pytest.approx(0.4)
    assert iou == pytest.approx(0.6)


def test_detect_handles_no_results_and_missing_boxes(monkeypatch: pytest.MonkeyPatch):
    md = _load_megadetector_module("wd_md_empty")
    monkeypatch.setattr(md, "_select_device", lambda: "cpu")

    class FakeYOLO:
        def __init__(self, _path: str):  # pragma: no cover - trivial
            pass

        def predict(self, img, **kwargs):
            return []

    monkeypatch.setattr(md, "YOLO", FakeYOLO)

    det = md.MegaDetector("/tmp/dummy.pt")
    assert det.detect(np.zeros((1, 1, 3), dtype=np.uint8)) == []

    # Now return a result with boxes=None
    class FakeYOLOB(FakeYOLO):
        def predict(self, img, **kwargs):
            return [FakeResult(None)]

    monkeypatch.setattr(md, "YOLO", FakeYOLOB)
    det = md.MegaDetector("/tmp/dummy.pt")
    assert det.detect(np.zeros((1, 1, 3), dtype=np.uint8)) == []
