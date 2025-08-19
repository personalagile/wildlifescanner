from __future__ import annotations

import types
from pathlib import Path

from wildlifescanner import pipeline as pipe
from wildlifescanner.models import VideoSegment


def _install_cv2_stub(monkeypatch):
    class _Cap:
        def isOpened(self):
            return True

        def read(self):
            return False, None  # end immediately

        def release(self):
            pass

    cv2 = types.SimpleNamespace(VideoCapture=lambda *_a, **_k: _Cap())
    monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)


def test_analyze_and_extract_postprocess_replaces_file(monkeypatch, tmp_path: Path):
    _install_cv2_stub(monkeypatch)

    # probe_video stub
    monkeypatch.setattr(pipe, "probe_video", lambda _p: (25.0, 0, 0.0))

    # segments stub (regardless of activity_times)
    seg = VideoSegment(start=0.0, end=1.0)
    monkeypatch.setattr(pipe, "compute_segments", lambda *_a, **_k: [seg])

    # extract_segments creates an output file
    def fake_extract(_inp, out_dir: Path, _segs, _logger):
        out = out_dir / "clip.mp4"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"orig")
        return [out]

    monkeypatch.setattr(pipe, "extract_segments", fake_extract)

    # postprocess writes tmp and returns True (changed)
    def fake_pp(input_video: Path, output_video: Path, **_k):  # type: ignore[no-untyped-def]
        output_video.write_bytes(b"processed")
        return True

    monkeypatch.setattr(pipe, "postprocess_zoom_and_tracking", fake_pp)

    logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)

    cfg = types.SimpleNamespace(
        output_dir=tmp_path,
        preroll_sec=0.0,
        postroll_sec=0.0,
        min_activity_sec=0.0,
        merge_gap_sec=0.0,
        frame_stride=1,
        zoom_enabled=True,
        tracking_enabled=False,
    )

    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"data")

    outs = pipe.analyze_and_extract(
        input_video,
        cfg,
        detector=types.SimpleNamespace(detect=lambda _f: []),
        logger=logger,
    )
    assert len(outs) == 1
    out_path = outs[0]
    # file was replaced with processed content
    assert out_path.read_bytes() == b"processed"


def test_analyze_and_extract_postprocess_cleans_tmp(monkeypatch, tmp_path: Path):
    _install_cv2_stub(monkeypatch)
    monkeypatch.setattr(pipe, "probe_video", lambda _p: (25.0, 0, 0.0))
    seg = VideoSegment(start=0.0, end=1.0)
    monkeypatch.setattr(pipe, "compute_segments", lambda *_a, **_k: [seg])

    def fake_extract(_inp, out_dir: Path, _segs, _logger):
        out = out_dir / "clip2.mp4"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"orig")
        return [out]

    monkeypatch.setattr(pipe, "extract_segments", fake_extract)

    # postprocess returns False and doesn't create tmp
    def fake_pp(input_video: Path, output_video: Path, **_k):  # type: ignore[no-untyped-def]
        # explicitly create tmp then return False to trigger cleanup
        output_video.write_bytes(b"tmp")
        return False

    monkeypatch.setattr(pipe, "postprocess_zoom_and_tracking", fake_pp)

    logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)

    cfg = types.SimpleNamespace(
        output_dir=tmp_path,
        preroll_sec=0.0,
        postroll_sec=0.0,
        min_activity_sec=0.0,
        merge_gap_sec=0.0,
        frame_stride=1,
        zoom_enabled=True,
        tracking_enabled=False,
    )

    input_video = tmp_path / "input2.mp4"
    input_video.write_bytes(b"data")

    outs = pipe.analyze_and_extract(
        input_video,
        cfg,
        detector=types.SimpleNamespace(detect=lambda _f: []),
        logger=logger,
    )
    assert len(outs) == 1
    out_path = outs[0]
    # original kept
    assert out_path.read_bytes() == b"orig"
    # tmp cleaned up
    tmp = out_path.with_name(out_path.stem + "_zoom" + out_path.suffix)
    assert not tmp.exists()


def test_analyze_and_extract_keep_postprocessed(monkeypatch, tmp_path: Path):
    _install_cv2_stub(monkeypatch)
    monkeypatch.setattr(pipe, "probe_video", lambda _p: (25.0, 0, 0.0))
    seg = VideoSegment(start=0.0, end=1.0)
    monkeypatch.setattr(pipe, "compute_segments", lambda *_a, **_k: [seg])

    def fake_extract(_inp, out_dir: Path, _segs, _logger):
        out = out_dir / "clip3.mp4"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"orig")
        return [out]

    monkeypatch.setattr(pipe, "extract_segments", fake_extract)

    def fake_pp(input_video: Path, output_video: Path, **_k):  # type: ignore[no-untyped-def]
        output_video.write_bytes(b"processed")
        return True

    monkeypatch.setattr(pipe, "postprocess_zoom_and_tracking", fake_pp)

    logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)

    cfg = types.SimpleNamespace(
        output_dir=tmp_path,
        preroll_sec=0.0,
        postroll_sec=0.0,
        min_activity_sec=0.0,
        merge_gap_sec=0.0,
        frame_stride=1,
        zoom_enabled=True,
        tracking_enabled=False,
        keep_postprocessed=True,
    )

    input_video = tmp_path / "input3.mp4"
    input_video.write_bytes(b"data")

    outs = pipe.analyze_and_extract(
        input_video,
        cfg,
        detector=types.SimpleNamespace(detect=lambda _f: []),
        logger=logger,
    )
    assert len(outs) == 1
    out_path = outs[0]
    # original kept, processed kept alongside
    assert out_path.read_bytes() == b"orig"
    tmp = out_path.with_name(out_path.stem + "_zoom" + out_path.suffix)
    assert tmp.exists()
    assert tmp.read_bytes() == b"processed"
