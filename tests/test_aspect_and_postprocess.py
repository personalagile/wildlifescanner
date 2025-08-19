from __future__ import annotations

import types
from pathlib import Path

import pytest

from wildlifescanner.config import AppConfig
from wildlifescanner.processing import video as video_mod


def test_expand_to_minimum_keeps_aspect_and_bounds():
    # frame 1920x1080, target 960x540 => aspect 16:9
    frame_w, frame_h = 1920, 1080
    min_w, min_h = 960, 540
    ar = min_w / min_h
    # small rect near center
    rect = (900, 500, 50, 40)
    x, y, w, h = video_mod._expand_to_minimum(
        rect, frame_w, frame_h, min_w, min_h, margin_ratio=0.1, target_aspect=ar
    )
    assert 0 <= x <= frame_w - 1
    assert 0 <= y <= frame_h - 1
    assert w > 0 and h > 0
    # locked aspect
    assert pytest.approx(w / h, rel=1e-3) == ar
    # inside frame
    assert x + w <= frame_w
    assert y + h <= frame_h


def test_expand_to_minimum_fits_when_larger_than_frame():
    frame_w, frame_h = 640, 360
    min_w, min_h = 1280, 720  # larger than frame
    ar = min_w / min_h
    rect = (0, 0, 10, 10)
    x, y, w, h = video_mod._expand_to_minimum(
        rect, frame_w, frame_h, min_w, min_h, margin_ratio=0.0, target_aspect=ar
    )
    # should be centered inside frame with same aspect, best fit
    assert (x, y) == ((frame_w - w) // 2, (frame_h - h) // 2)
    assert w <= frame_w and h <= frame_h
    assert pytest.approx(w / h, rel=1e-3) == ar


def test_static_crop_ffmpeg_builds_scale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {"vf": None}

    class _Result:
        def __init__(self):
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        # find -vf parameter value
        if "-vf" in cmd:
            idx = cmd.index("-vf")
            captured["vf"] = cmd[idx + 1]
        return _Result()

    monkeypatch.setattr(video_mod.subprocess, "run", fake_run)
    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    inp.write_bytes(b"fake")
    # crop any rect and scale to exact 960x540
    video_mod._static_crop_ffmpeg(inp, out, (10, 20, 300, 200), 960, 540)
    assert captured["vf"] is not None
    assert "scale=960:540" in captured["vf"]


def _install_cv2_stub_for_postprocess(monkeypatch: pytest.MonkeyPatch, *, w=1280, h=720, frames=3):
    class _Cap:
        def __init__(self):
            self._opened = True
            self._i = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 3:
                return w
            if prop == 4:
                return h
            if prop == 5:
                return 25.0
            if prop == 7:
                return frames
            return 0

        def read(self):
            if self._i >= frames:
                return False, None
            self._i += 1
            import numpy as np  # type: ignore

            return True, np.zeros((h, w, 3), dtype=np.uint8)

        def release(self):
            pass

    cv2 = types.SimpleNamespace()
    cv2.VideoCapture = lambda *_a, **_k: _Cap()
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_COUNT = 7
    monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)


def test_postprocess_zoom_static_calls_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_cv2_stub_for_postprocess(monkeypatch)

    # detector: one bbox on first frame
    class D:  # simple detection object with coords
        def __init__(self, x1, y1, x2, y2):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    calls = {"run": 0}

    def fake_run(_cmd, *a, **k):  # type: ignore[no-untyped-def]
        calls["run"] += 1

        class R:
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(video_mod.subprocess, "run", fake_run)

    cfg = AppConfig(input_dir=tmp_path, output_dir=tmp_path)
    cfg.zoom_enabled = True
    cfg.tracking_enabled = False
    cfg.min_output_width = 960
    cfg.min_output_height = 540
    cfg.frame_stride = 1

    # detector returns one bbox then none
    det_calls = {"i": 0}

    def detect(_frame):  # type: ignore[no-untyped-def]
        det_calls["i"] += 1
        return [D(100, 100, 300, 260)] if det_calls["i"] == 1 else []

    ok = video_mod.postprocess_zoom_and_tracking(
        input_video=tmp_path / "in.mp4",
        output_video=tmp_path / "out.mp4",
        cfg=cfg,
        detector=types.SimpleNamespace(detect=detect),
        logger=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    assert ok is True
    assert calls["run"] >= 1  # ffmpeg invoked


auto_called = {"dyn": 0}


def test_postprocess_tracking_calls_dynamic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_cv2_stub_for_postprocess(monkeypatch)

    def fake_dyn(*_a, **_k):  # type: ignore[no-untyped-def]
        auto_called["dyn"] += 1
        return None

    monkeypatch.setattr(video_mod, "_dynamic_crop_cv2", fake_dyn)

    cfg = AppConfig(input_dir=tmp_path, output_dir=tmp_path)
    cfg.zoom_enabled = False
    cfg.tracking_enabled = True
    cfg.min_output_width = 960
    cfg.min_output_height = 540

    ok = video_mod.postprocess_zoom_and_tracking(
        input_video=tmp_path / "in.mp4",
        output_video=tmp_path / "out.mp4",
        cfg=cfg,
        detector=types.SimpleNamespace(detect=lambda _f: []),
        logger=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    assert ok is True
    assert auto_called["dyn"] == 1
