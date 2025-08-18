from __future__ import annotations

import logging
from pathlib import Path

import pytest

from wildlifescanner.models import VideoSegment
from wildlifescanner.processing import video as video_mod


def test_probe_duration_ffmpeg_ok(monkeypatch: pytest.MonkeyPatch):
    class _Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def _run(_cmd, *args, **kwargs):
        return _Result(stdout='{"streams": [{"codec_type": "video", "duration": "1.23"}]}')

    monkeypatch.setattr(video_mod.subprocess, "run", _run)
    assert video_mod._probe_duration_ffmpeg(Path("x.mp4")) == pytest.approx(1.23)


def test_probe_duration_ffmpeg_exception(monkeypatch: pytest.MonkeyPatch):
    def _run(_cmd, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(video_mod.subprocess, "run", _run)
    assert video_mod._probe_duration_ffmpeg(Path("x.mp4")) == 0.0


def test_extract_segments_copy_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, int] = {"copy": 0, "re": 0}

    def _copy(_inp, _out, _seg):
        calls["copy"] += 1

    def _re(_inp, _out, _seg):
        calls["re"] += 1

    monkeypatch.setattr(video_mod, "_cut_stream_copy", _copy)
    monkeypatch.setattr(video_mod, "_cut_reencode", _re)

    logger = logging.getLogger("test")
    segs = [VideoSegment(0.0, 0.5), VideoSegment(1.0, 2.0)]
    outs = video_mod.extract_segments(Path("in.mp4"), tmp_path, segs, logger)
    assert len(outs) == 2
    assert calls["copy"] == 2
    assert calls["re"] == 0


def test_extract_segments_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, int] = {"copy": 0, "re": 0}

    def _copy(_inp, _out, _seg):
        calls["copy"] += 1
        raise RuntimeError("fail")

    def _re(_inp, _out, _seg):
        calls["re"] += 1

    monkeypatch.setattr(video_mod, "_cut_stream_copy", _copy)
    monkeypatch.setattr(video_mod, "_cut_reencode", _re)

    logger = logging.getLogger("test")
    segs = [VideoSegment(0.0, 0.5)]
    outs = video_mod.extract_segments(Path("in.mp4"), tmp_path, segs, logger)
    assert len(outs) == 1
    assert calls["copy"] == 1
    assert calls["re"] == 1
