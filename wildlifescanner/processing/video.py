from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from ..models import VideoSegment


def probe_video(path: Path) -> Tuple[float, int, float]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else _probe_duration_ffmpeg(path)
    cap.release()
    return float(fps), frame_count, float(duration)


def _probe_duration_ffmpeg(path: Path) -> float:
    try:
        import ffmpeg  # type: ignore
        info = ffmpeg.probe(str(path))
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                dur = s.get("duration") or info.get("format", {}).get("duration")
                if dur is not None:
                    return float(dur)
    except Exception:
        pass
    return 0.0


def format_segment_filename(base: str, idx: int, seg: VideoSegment, ext: str) -> str:
    s_ms = int(round(seg.start * 1000))
    e_ms = int(round(seg.end * 1000))
    return f"{base}_seg{idx:03d}_{s_ms}ms_{e_ms}ms{ext}"


def extract_segments(
    input_video: Path,
    output_dir: Path,
    segments: Iterable[VideoSegment],
    logger: logging.Logger,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = input_video.stem
    ext = input_video.suffix or ".mp4"
    paths: List[Path] = []
    for i, seg in enumerate(segments, start=1):
        out_path = output_dir / format_segment_filename(base, i, seg, ext)
        try:
            _cut_stream_copy(input_video, out_path, seg)
            logger.info(f"Wrote segment (copy): {out_path}")
        except Exception as e:
            logger.warning(f"Stream copy failed, falling back to re-encode: {e}")
            _cut_reencode(input_video, out_path, seg)
            logger.info(f"Wrote segment (re-encode): {out_path}")
        paths.append(out_path)
    return paths


def _cut_stream_copy(inp: Path, out: Path, seg: VideoSegment) -> None:
    import ffmpeg  # type: ignore
    start = max(0.0, seg.start)
    dur = max(0.0, seg.end - seg.start)
    (
        ffmpeg
        .input(str(inp), ss=start)
        .output(str(out), t=dur, c="copy", movflags="faststart")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def _cut_reencode(inp: Path, out: Path, seg: VideoSegment) -> None:
    import ffmpeg  # type: ignore
    start = max(0.0, seg.start)
    dur = max(0.0, seg.end - seg.start)
    (
        ffmpeg
        .input(str(inp), ss=start)
        .output(
            str(out),
            t=dur,
            vcodec="libx264",
            acodec="aac",
            preset="veryfast",
            crf=22,
            movflags="faststart",
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
