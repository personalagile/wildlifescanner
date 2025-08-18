from __future__ import annotations

import logging
import types
from pathlib import Path

import numpy as np
import pytest

from wildlifescanner.config import AppConfig
from wildlifescanner.models import VideoSegment
from wildlifescanner.pipeline import analyze_and_extract


class _CapStub:
    def __init__(self, frames: int, opened: bool = True) -> None:
        self._opened = opened
        self._frames = frames
        self._read_idx = 0

    def isOpened(self) -> bool:
        return self._opened

    def get(self, _prop):  # minimal API compatibility
        return 0

    def read(self):
        if not self._opened:
            return False, None
        if self._read_idx >= self._frames:
            return False, None
        self._read_idx += 1
        # Return a dummy BGR frame
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


@pytest.fixture()
def cfg(tmp_path: Path) -> AppConfig:
    out = tmp_path / "out"
    return AppConfig(input_dir=tmp_path, output_dir=out)


def _install_cv2_stub(monkeypatch: pytest.MonkeyPatch, cap: _CapStub) -> None:
    cv2 = types.SimpleNamespace()
    cv2.VideoCapture = lambda *_args, **_kwargs: cap
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_COUNT = 7
    monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)


def test_analyze_and_extract_cannot_open(
    tmp_path: Path, cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
):
    # Probe returns some values
    monkeypatch.setattr("wildlifescanner.pipeline.probe_video", lambda _p: (10.0, 100, 10.0))
    # cv2 stub that cannot open
    _install_cv2_stub(monkeypatch, _CapStub(frames=0, opened=False))

    logger = logging.getLogger("test")
    with pytest.raises(RuntimeError):
        analyze_and_extract(
            tmp_path / "vid.mp4",
            cfg,
            detector=types.SimpleNamespace(detect=lambda f: []),
            logger=logger,
        )


def test_analyze_and_extract_no_segments(
    tmp_path: Path, cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("wildlifescanner.pipeline.probe_video", lambda _p: (10.0, 10, 1.0))
    _install_cv2_stub(monkeypatch, _CapStub(frames=3, opened=True))

    # Detector finds nothing
    det = types.SimpleNamespace(detect=lambda f: [])
    # Force no segments
    monkeypatch.setattr("wildlifescanner.pipeline.compute_segments", lambda *a, **k: [])

    logger = logging.getLogger("test")
    outs = analyze_and_extract(tmp_path / "vid.mp4", cfg, det, logger)
    assert outs == []


def test_analyze_and_extract_with_segments(
    tmp_path: Path, cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
):
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("wildlifescanner.pipeline.probe_video", lambda _p: (10.0, 10, 1.0))
    _install_cv2_stub(monkeypatch, _CapStub(frames=3, opened=True))

    # Detector triggers activity on first frame
    calls = {"i": 0}

    def _detect(_frame):
        calls["i"] += 1
        return [object()] if calls["i"] == 1 else []

    det = types.SimpleNamespace(detect=_detect)

    # Return a deterministic segment and capture the call to extract_segments
    segs = [VideoSegment(0.0, 0.5)]
    monkeypatch.setattr("wildlifescanner.pipeline.compute_segments", lambda *a, **k: segs)

    expected = [out_dir / "vid_seg001_0ms_500ms.mp4"]
    monkeypatch.setattr(
        "wildlifescanner.pipeline.extract_segments",
        lambda video, out, s, _logger: expected,
    )

    logger = logging.getLogger("test")
    outs = analyze_and_extract(tmp_path / "vid.mp4", cfg, det, logger)
    assert outs == expected
