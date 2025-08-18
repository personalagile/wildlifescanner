from __future__ import annotations

import types
from pathlib import Path

import pytest

from wildlifescanner.models import VideoSegment
from wildlifescanner.processing import video as video_mod


class _CV2Cap:
    def __init__(self, opened: bool, fps: float, frames: int) -> None:
        self._opened = opened
        self._fps = fps
        self._frames = frames

    def isOpened(self) -> bool:  # OpenCV API
        return self._opened

    def get(self, prop):  # OpenCV API
        if prop == self._cv2.CAP_PROP_FPS:
            return self._fps
        if prop == self._cv2.CAP_PROP_FRAME_COUNT:
            return self._frames
        return 0

    def release(self) -> None:
        pass

    # cv2 constants container gets injected later
    _cv2: types.SimpleNamespace


def _install_cv2_stub(
    monkeypatch: pytest.MonkeyPatch,
    opened: bool,
    fps: float,
    frames: int,
) -> None:
    cv2 = types.SimpleNamespace()
    cap = _CV2Cap(opened=opened, fps=fps, frames=frames)
    cap._cv2 = cv2  # provide constants namespace
    cv2.VideoCapture = lambda *_a, **_k: cap
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_COUNT = 7
    monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)


def test_probe_video_cv2_success(monkeypatch: pytest.MonkeyPatch):
    _install_cv2_stub(monkeypatch, opened=True, fps=50.0, frames=250)
    fps, count, dur = video_mod.probe_video(Path("x.mp4"))
    assert fps == 50.0
    assert count == 250
    assert dur == pytest.approx(5.0)


def test_probe_video_cv2_ffmpeg_fallback(monkeypatch: pytest.MonkeyPatch):
    # frame_count will be 0 -> fallback to ffmpeg duration
    _install_cv2_stub(monkeypatch, opened=True, fps=0.0, frames=0)

    class _FF:
        @staticmethod
        def probe(_p: str):
            return {"streams": [{"codec_type": "video", "duration": "1.5"}]}

    monkeypatch.setitem(__import__("sys").modules, "ffmpeg", _FF)
    fps, count, dur = video_mod.probe_video(Path("x.mp4"))
    assert fps == 25.0  # default when cv2 fps is 0
    assert count == 0
    assert dur == pytest.approx(1.5)


def test_cut_stream_copy_invokes_ffmpeg_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: dict[str, dict] = {"args": {}}

    class _Node:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def output(self, out, t=None, c=None, movflags=None):
            calls["args"]["output"] = {"out": out, "t": t, "c": c, "movflags": movflags}
            return self

        def overwrite_output(self):
            calls["args"]["overwrite"] = True
            return self

        def run(self, capture_stdout=False, capture_stderr=False):
            calls["args"]["run"] = {"stdout": capture_stdout, "stderr": capture_stderr}
            return None

    class _FF:
        @staticmethod
        def input(inp, ss=None):
            calls["args"]["input"] = {"inp": inp, "ss": ss}
            return _Node("input")

    monkeypatch.setitem(__import__("sys").modules, "ffmpeg", _FF)

    seg = VideoSegment(1.0, 3.5)
    video_mod._cut_stream_copy(Path("in.mp4"), tmp_path / "o.mp4", seg)

    assert calls["args"]["input"]["ss"] == pytest.approx(1.0)
    assert calls["args"]["output"]["t"] == pytest.approx(2.5)
    assert calls["args"]["output"]["c"] == "copy"
    assert calls["args"]["output"]["movflags"] == "faststart"
    assert calls["args"].get("overwrite") is True
    assert calls["args"]["run"] == {"stdout": True, "stderr": True}


def test_cut_reencode_invokes_ffmpeg_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: dict[str, dict] = {"args": {}}

    class _Node:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def output(
            self,
            out,
            t=None,
            vcodec=None,
            acodec=None,
            preset=None,
            crf=None,
            movflags=None,
        ):
            calls["args"]["output"] = {
                "out": out,
                "t": t,
                "vcodec": vcodec,
                "acodec": acodec,
                "preset": preset,
                "crf": crf,
                "movflags": movflags,
            }
            return self

        def overwrite_output(self):
            calls["args"]["overwrite"] = True
            return self

        def run(self, capture_stdout=False, capture_stderr=False):
            calls["args"]["run"] = {"stdout": capture_stdout, "stderr": capture_stderr}
            return None

    class _FF:
        @staticmethod
        def input(inp, ss=None):
            calls["args"]["input"] = {"inp": inp, "ss": ss}
            return _Node("input")

    monkeypatch.setitem(__import__("sys").modules, "ffmpeg", _FF)

    seg = VideoSegment(2.0, 2.5)
    video_mod._cut_reencode(Path("in.mp4"), tmp_path / "o.mp4", seg)

    assert calls["args"]["input"]["ss"] == pytest.approx(2.0)
    assert calls["args"]["output"]["t"] == pytest.approx(0.5)
    assert calls["args"]["output"]["vcodec"] == "libx264"
    assert calls["args"]["output"]["acodec"] == "aac"
    assert calls["args"]["output"]["preset"] == "veryfast"
    assert calls["args"]["output"]["crf"] == 22
    assert calls["args"]["output"]["movflags"] == "faststart"
    assert calls["args"].get("overwrite") is True
    assert calls["args"]["run"] == {"stdout": True, "stderr": True}
