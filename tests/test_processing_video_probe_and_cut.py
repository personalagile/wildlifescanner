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

    # stub subprocess.run used by _probe_duration_ffmpeg to emulate ffprobe json
    calls: dict[str, object] = {}

    class _Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def _run(cmd, *args, **kwargs):
        calls["cmd"] = cmd
        return _Result(stdout='{"streams": [{"codec_type": "video", "duration": "1.5"}]}')

    monkeypatch.setattr(video_mod.subprocess, "run", _run)
    fps, count, dur = video_mod.probe_video(Path("x.mp4"))
    assert fps == 25.0  # default when cv2 fps is 0
    assert count == 0
    assert dur == pytest.approx(1.5)


def test_cut_stream_copy_invokes_ffmpeg_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, list[str]] = {}

    class _Result:
        def __init__(self) -> None:
            self.stdout = ""
            self.stderr = ""

    def _run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(video_mod.subprocess, "run", _run)

    seg = VideoSegment(1.0, 3.5)
    inp = Path("in.mp4")
    out = tmp_path / "o.mp4"
    video_mod._cut_stream_copy(inp, out, seg)

    assert captured["cmd"] == [
        "ffmpeg",
        "-ss",
        "1.0",
        "-i",
        str(inp),
        "-t",
        "2.5",
        "-c",
        "copy",
        "-movflags",
        "faststart",
        "-y",
        str(out),
    ]


def test_cut_reencode_invokes_ffmpeg_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, list[str]] = {}

    class _Result:
        def __init__(self) -> None:
            self.stdout = ""
            self.stderr = ""

    def _run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(video_mod.subprocess, "run", _run)

    seg = VideoSegment(2.0, 2.5)
    inp = Path("in.mp4")
    out = tmp_path / "o.mp4"
    video_mod._cut_reencode(inp, out, seg)

    assert captured["cmd"] == [
        "ffmpeg",
        "-ss",
        "2.0",
        "-i",
        str(inp),
        "-t",
        "0.5",
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-movflags",
        "faststart",
        "-y",
        str(out),
    ]
