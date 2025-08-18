from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _torch_stub(mps_avail=False, cuda_avail=False, raise_on_backends=False):
    if raise_on_backends:

        class _B:
            def __getattr__(self, _):
                raise RuntimeError("boom")

        backends = _B()
    else:
        backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps_avail))
    cuda = types.SimpleNamespace(is_available=lambda: cuda_avail)
    return types.SimpleNamespace(backends=backends, cuda=cuda)


def _ultra_stub(names_source: str = "dict", predict_result=None):
    # names_source: "dict" -> instance has names dict; "list" -> instance.model.names list
    class _YOLO:
        def __init__(self, *_a, **_k):
            if names_source == "dict":
                self.names = {"0": "bird", "1": "cat"}
            else:
                self.model = types.SimpleNamespace(names=["bird", "cat"])  # list path

        def predict(self, *a, **k):
            return [] if predict_result is None else predict_result

    return types.SimpleNamespace(YOLO=_YOLO)


def _load_yolo(torch_stub, ultra_stub):
    sys.modules["torch"] = torch_stub
    sys.modules["ultralytics"] = ultra_stub
    # Force fresh import of module under test
    sys.modules.pop("wildlifescanner.detectors.yolo", None)
    mod = importlib.import_module("wildlifescanner.detectors.yolo")
    return mod


def test_select_device_mps_preferred():
    yolo = _load_yolo(_torch_stub(mps_avail=True, cuda_avail=True), _ultra_stub())
    assert yolo._select_device() == "mps"


def test_select_device_cuda_when_no_mps():
    yolo = _load_yolo(_torch_stub(mps_avail=False, cuda_avail=True), _ultra_stub())
    assert yolo._select_device() == "cuda"


def test_select_device_cpu_when_none():
    yolo = _load_yolo(_torch_stub(mps_avail=False, cuda_avail=False), _ultra_stub())
    assert yolo._select_device() == "cpu"


def test_select_device_exception_falls_back_cpu():
    yolo = _load_yolo(_torch_stub(raise_on_backends=True), _ultra_stub())
    assert yolo._select_device() == "cpu"


def test_init_class_names_from_dict():
    yolo = _load_yolo(_torch_stub(), _ultra_stub(names_source="dict"))
    det = yolo.YOLODetector(model_path="x.pt")
    assert det.class_names[0] == "bird" and det.class_names[1] == "cat"


def test_init_class_names_from_list():
    yolo = _load_yolo(_torch_stub(), _ultra_stub(names_source="list"))
    det = yolo.YOLODetector(model_path="x.pt")
    assert det.class_names[0] == "bird" and det.class_names[1] == "cat"


def _boxes(xyxy=True, conf=True, cls=True):
    class _Arr:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

    b = types.SimpleNamespace()
    if xyxy:
        b.xyxy = _Arr([[1, 2, 3, 4], [10, 20, 30, 40]])
    if conf:
        b.conf = _Arr([0.9, 0.8])
    if cls:
        b.cls = _Arr([0, 1])
    return b


def test_detect_empty_results():
    results = []
    yolo = _load_yolo(_torch_stub(), _ultra_stub(predict_result=results))
    det = yolo.YOLODetector()
    out = det.detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert out == []


def test_detect_no_boxes():
    results = [types.SimpleNamespace(boxes=None)]
    yolo = _load_yolo(_torch_stub(), _ultra_stub(predict_result=results))
    det = yolo.YOLODetector()
    out = det.detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert out == []


@pytest.mark.parametrize("missing", ["xyxy", "conf", "cls"])
def test_detect_missing_arrays_returns_empty(missing):
    bx = _boxes(xyxy=(missing != "xyxy"), conf=(missing != "conf"), cls=(missing != "cls"))
    results = [types.SimpleNamespace(boxes=bx)]
    yolo = _load_yolo(_torch_stub(), _ultra_stub(predict_result=results))
    det = yolo.YOLODetector()
    out = det.detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert out == []


def test_detect_filters_allowed_classes():
    bx = _boxes()
    results = [types.SimpleNamespace(boxes=bx)]
    yolo = _load_yolo(_torch_stub(), _ultra_stub(predict_result=results))
    det = yolo.YOLODetector(allowed_classes=("bird",))
    out = det.detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert len(out) == 1
    det0 = out[0]
    assert det0.cls_name == "bird" and det0.x1 == pytest.approx(1.0)
