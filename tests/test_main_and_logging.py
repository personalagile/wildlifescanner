from __future__ import annotations

import contextlib
import types
from pathlib import Path

from wildlifescanner import main as main_mod
from wildlifescanner.logging_setup import setup_logging


def test_setup_logging_idempotent(tmp_path: Path):
    logger1 = setup_logging(tmp_path, level="INFO")
    # 2 handlers: console + file
    assert len(logger1.handlers) == 2
    logger2 = setup_logging(tmp_path, level="DEBUG")
    # No duplicate handlers added, returns same logger
    assert logger1 is logger2
    assert len(logger2.handlers) == 2


def test__process_video_closes_detector_on_success(monkeypatch, tmp_path: Path):
    closed = {"yes": False}

    class Det:
        def __init__(self):
            self.closed = False

        def close(self):
            closed["yes"] = True

    def fake_create(*_a, **_k):  # type: ignore[no-untyped-def]
        return Det()

    outputs = {"paths": [tmp_path / "x.mp4"]}

    def fake_analyze(_p, _cfg, _det, _logger):  # type: ignore[no-untyped-def]
        return outputs["paths"]

    monkeypatch.setattr(main_mod, "create_detector", fake_create)
    monkeypatch.setattr(main_mod, "analyze_and_extract", fake_analyze)

    cfg = types.SimpleNamespace(
        detector="YOLO",
        yolo_model="m.pt",
        confidence_threshold=0.3,
        nms_iou=0.5,
        animal_classes=("cat",),
        output_dir=tmp_path,
        log_level="INFO",
        input_dir=tmp_path,
    )
    main_mod._process_video(
        tmp_path / "a.mp4",
        cfg,
        logger=types.SimpleNamespace(info=lambda *a, **k: None),
    )
    assert closed["yes"] is True


def test__process_video_closes_detector_on_error(monkeypatch, tmp_path: Path):
    closed = {"yes": False}

    class Det:
        def close(self):
            closed["yes"] = True

    def fake_create(*_a, **_k):  # type: ignore[no-untyped-def]
        return Det()

    def fake_analyze(_p, _cfg, _det, _logger):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod, "create_detector", fake_create)
    monkeypatch.setattr(main_mod, "analyze_and_extract", fake_analyze)

    cfg = types.SimpleNamespace(
        detector="YOLO",
        yolo_model="m.pt",
        confidence_threshold=0.3,
        nms_iou=0.5,
        animal_classes=("cat",),
        output_dir=tmp_path,
        log_level="INFO",
        input_dir=tmp_path,
    )
    # _process_video propagates exceptions; ensure detector is closed regardless
    with contextlib.suppress(RuntimeError):
        main_mod._process_video(
            tmp_path / "a.mp4",
            cfg,
            logger=types.SimpleNamespace(info=lambda *a, **k: None),
        )
    assert closed["yes"] is True
