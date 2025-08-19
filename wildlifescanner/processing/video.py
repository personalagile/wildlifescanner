from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path

from ..config import AppConfig
from ..detectors.base import AnimalDetector
from ..models import VideoSegment


def probe_video(path: Path) -> tuple[float, int, float]:
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
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(result.stdout or "{}")
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                dur = s.get("duration") or info.get("format", {}).get("duration")
                if dur is not None:
                    return float(dur)
        dur = info.get("format", {}).get("duration")
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
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = input_video.stem
    ext = input_video.suffix or ".mp4"
    paths: list[Path] = []
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
    start = max(0.0, seg.start)
    dur = max(0.0, seg.end - seg.start)
    cmd = [
        "ffmpeg",
        "-ss",
        f"{start}",
        "-i",
        str(inp),
        "-t",
        f"{dur}",
        "-c",
        "copy",
        "-movflags",
        "faststart",
        "-y",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# --- Zoom/Tracking post-processing -------------------------------------------------


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _compute_union_bbox(
    cap, detector: AnimalDetector, stride: int
) -> tuple[int, int, int, int] | None:
    """
    Iterate frames and compute union bbox across all detections in the clip.
    Returns (x, y, w, h) in integer pixels, or None if no detections.
    """
    w = int(cap.get(3))  # CAP_PROP_FRAME_WIDTH
    h = int(cap.get(4))  # CAP_PROP_FRAME_HEIGHT
    has_any = False
    x1u, y1u, x2u, y2u = w, h, 0, 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        # Ensure BGR frame shape is valid
        if frame is None or frame.size == 0:
            idx += 1
            continue
        dets = detector.detect(frame)
        if dets:
            for d in dets:
                x1u = min(x1u, int(d.x1))
                y1u = min(y1u, int(d.y1))
                x2u = max(x2u, int(d.x2))
                y2u = max(y2u, int(d.y2))
            has_any = True
        idx += 1
    if not has_any:
        return None
    x1u = _clamp(x1u, 0, w - 1)
    y1u = _clamp(y1u, 0, h - 1)
    x2u = _clamp(x2u, x1u + 1, w)
    y2u = _clamp(y2u, y1u + 1, h)
    return x1u, y1u, x2u - x1u, y2u - y1u


def _expand_to_minimum(
    rect: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    min_w: int,
    min_h: int,
    margin_ratio: float = 0.1,
    target_aspect: float | None = None,
) -> tuple[int, int, int, int]:
    """
    Expand a rectangle by margin_ratio and to at least min_w/min_h while staying inside the frame.
    Returns (x, y, w, h).
    """
    x, y, w, h = rect
    # add margin
    add = int(round(max(w, h) * margin_ratio))
    w2, h2 = max(1, w + 2 * add), max(1, h + 2 * add)
    cx, cy = x + w // 2, y + h // 2

    # ensure minimums
    w2 = max(w2, min_w)
    h2 = max(h2, min_h)

    # lock aspect if requested by expanding the smaller side
    if target_aspect and target_aspect > 0:
        cur_ar = w2 / h2
        if cur_ar < target_aspect:  # too tall; increase width
            w2 = int(round(h2 * target_aspect))
        elif cur_ar > target_aspect:  # too wide; increase height
            h2 = int(round(w2 / target_aspect))

    # place centered
    x2 = cx - w2 // 2
    y2 = cy - h2 // 2

    # clamp to frame; if crop exceeds frame, fit inside while keeping aspect
    x2 = _clamp(x2, 0, max(0, frame_w - w2))
    y2 = _clamp(y2, 0, max(0, frame_h - h2))

    # adjust if crop exceeds frame due to rounding or expansion
    if x2 + w2 > frame_w:
        x2 = max(0, frame_w - w2)
    if y2 + h2 > frame_h:
        y2 = max(0, frame_h - h2)

    # If still larger than frame (very tight scenes), fit a centered rect with target aspect
    if w2 > frame_w or h2 > frame_h:
        if target_aspect and target_aspect > 0:
            # fit by width first
            fw, fh = frame_w, frame_h
            w_fit = fw
            h_fit = int(round(w_fit / target_aspect))
            if h_fit > fh:
                h_fit = fh
                w_fit = int(round(h_fit * target_aspect))
            w2, h2 = max(1, w_fit), max(1, h_fit)
            x2 = (fw - w2) // 2
            y2 = (fh - h2) // 2
        else:
            x2, y2, w2, h2 = 0, 0, frame_w, frame_h
    return int(x2), int(y2), int(w2), int(h2)


def _static_crop_ffmpeg(
    inp: Path,
    out: Path,
    crop: tuple[int, int, int, int],
    out_w: int,
    out_h: int,
) -> None:
    x, y, w, h = crop
    # Compose filter: crop with locked aspect and scale to exact stable output size (no distortion)
    vf = f"crop={w}:{h}:{x}:{y},scale={out_w}:{out_h}"
    cmd = [
        "ffmpeg",
        "-i",
        str(inp),
        "-vf",
        vf,
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
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _dynamic_crop_cv2(  # pragma: no cover (heavy video path; covered indirectly)
    inp: Path,
    out: Path,
    detector: AnimalDetector,
    min_w: int,
    min_h: int,
    stride: int,
    center_alpha: float,
    size_alpha: float,
    max_move_frac: float,
    max_zoom_frac: float,
    center_deadzone_frac: float,
    zoom_deadzone_frac: float,
    margin_dynamic: float,
) -> None:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Choose constant output size: at least min_w x min_h, fixed aspect
    out_w, out_h = int(max(min_w, 2)), int(max(min_h, 2))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter for dynamic crop")

    idx = 0
    # Camera smoothing parameters come from args (cfg)
    # center_alpha: easing factor for center movement per frame
    # size_alpha: easing factor for zoom (size) per frame
    # max_move_frac/max_zoom_frac: per-frame clamps
    # margin_dynamic: extra margin around subjects to avoid tight jitter
    # center_deadzone_frac/zoom_deadzone_frac: deadzone thresholds

    # Current camera rectangle and current target rectangle
    cam_rect: tuple[int, int, int, int] | None = None
    tgt_rect: tuple[int, int, int, int] | None = None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Update target on stride frames
            if idx % stride == 0:
                dets = detector.detect(frame)
                if dets:
                    x1 = frame_w
                    y1 = frame_h
                    x2 = 0
                    y2 = 0
                    for d in dets:
                        x1 = min(x1, int(d.x1))
                        y1 = min(y1, int(d.y1))
                        x2 = max(x2, int(d.x2))
                        y2 = max(y2, int(d.y2))
                    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                    tgt_rect = _expand_to_minimum(
                        rect,
                        frame_w,
                        frame_h,
                        min_w,
                        min_h,
                        margin_ratio=margin_dynamic,
                        target_aspect=(out_w / out_h),
                    )

            # Initialize camera rect if needed
            if cam_rect is None:
                cam_rect = tgt_rect if tgt_rect is not None else (0, 0, frame_w, frame_h)

            # Smoothly move camera toward target if available
            if tgt_rect is not None and cam_rect is not None:
                px, py, pw, ph = cam_rect
                tx, ty, tw, th = tgt_rect

                # centers
                cxp, cyp = px + pw // 2, py + ph // 2
                cxt, cyt = tx + tw // 2, ty + th // 2

                # deltas with deadzone and per-frame clamp
                max_dx = int(round(frame_w * max_move_frac))
                max_dy = int(round(frame_h * max_move_frac))
                dzx = int(round(frame_w * center_deadzone_frac))
                dzy = int(round(frame_h * center_deadzone_frac))
                dcx = cxt - cxp
                dcy = cyt - cyp
                # apply deadzone on center
                dcx = 0 if abs(dcx) <= dzx else dcx - (dzx if dcx > 0 else -dzx)
                dcy = 0 if abs(dcy) <= dzy else dcy - (dzy if dcy > 0 else -dzy)
                if dcx > max_dx:
                    dcx = max_dx
                elif dcx < -max_dx:
                    dcx = -max_dx
                if dcy > max_dy:
                    dcy = max_dy
                elif dcy < -max_dy:
                    dcy = -max_dy

                # eased step
                cxn = int(round(cxp + center_alpha * dcx))
                cyn = int(round(cyp + center_alpha * dcy))

                # size changes with deadzone and clamp
                max_dw = int(round(frame_w * max_zoom_frac))
                max_dh = int(round(frame_h * max_zoom_frac))
                zzw = int(round(frame_w * zoom_deadzone_frac))
                zzh = int(round(frame_h * zoom_deadzone_frac))
                dw = tw - pw
                dh = th - ph
                dw = 0 if abs(dw) <= zzw else dw - (zzw if dw > 0 else -zzw)
                dh = 0 if abs(dh) <= zzh else dh - (zzh if dh > 0 else -zzh)
                if dw > max_dw:
                    dw = max_dw
                elif dw < -max_dw:
                    dw = -max_dw
                if dh > max_dh:
                    dh = max_dh
                elif dh < -max_dh:
                    dh = -max_dh
                wn = int(round(pw + size_alpha * dw))
                hn = int(round(ph + size_alpha * dh))

                cam_rect = _expand_to_minimum(
                    (cxn - wn // 2, cyn - hn // 2, max(1, wn), max(1, hn)),
                    frame_w,
                    frame_h,
                    min_w,
                    min_h,
                    0.0,
                    target_aspect=(out_w / out_h),
                )

            # Crop and scale to output
            x, y, w, h = cam_rect if cam_rect is not None else (0, 0, frame_w, frame_h)
            x = _clamp(x, 0, max(0, frame_w - 1))
            y = _clamp(y, 0, max(0, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            roi = frame[y : y + h, x : x + w]
            if (w, h) != (out_w, out_h):
                # Resize is safe because crop aspect equals output aspect
                roi = cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            writer.write(roi)
            idx += 1
    finally:
        cap.release()
        writer.release()


def postprocess_zoom_and_tracking(
    input_video: Path,
    output_video: Path,
    cfg: AppConfig,
    detector: AnimalDetector,
    logger: logging.Logger,
) -> bool:
    """
    Apply zoom/tracking according to cfg. Returns True if a processed file was written.
    - If cfg.tracking_enabled: dynamic crop per frame (OpenCV re-encode)
    - Else if cfg.zoom_enabled: static crop based on union bbox (FFmpeg crop)
    - If no detections found: return False
    """
    if not cfg.zoom_enabled and not cfg.tracking_enabled:
        return False

    # lazy import cv2 only if needed
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        logger.warning("Cannot open video for zoom/tracking: %s", input_video)
        return False

    stride = max(1, int(cfg.frame_stride))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if cfg.tracking_enabled:
        cap.release()  # dynamic path re-opens internally
        _dynamic_crop_cv2(
            input_video,
            output_video,
            detector,
            cfg.min_output_width,
            cfg.min_output_height,
            stride,
            cfg.tracking_center_alpha,
            cfg.tracking_size_alpha,
            cfg.tracking_max_move_frac,
            cfg.tracking_max_zoom_frac,
            cfg.tracking_center_deadzone_frac,
            cfg.tracking_zoom_deadzone_frac,
            cfg.tracking_margin,
        )
        logger.info("Wrote tracked/zoomed segment: %s", output_video)
        return True

    # Static crop path (zoom only)
    bbox = _compute_union_bbox(cap, detector, stride)
    cap.release()
    if bbox is None:
        logger.info("Zoom requested but no animals detected in segment: %s", input_video)
        return False
    ar = cfg.min_output_width / cfg.min_output_height if cfg.min_output_height else None
    crop = _expand_to_minimum(
        bbox,
        frame_w,
        frame_h,
        cfg.min_output_width,
        cfg.min_output_height,
        target_aspect=ar,
    )
    _static_crop_ffmpeg(
        input_video,
        output_video,
        crop,
        cfg.min_output_width,
        cfg.min_output_height,
    )
    logger.info("Wrote zoomed segment: %s", output_video)
    return True


def _cut_reencode(inp: Path, out: Path, seg: VideoSegment) -> None:
    start = max(0.0, seg.start)
    dur = max(0.0, seg.end - seg.start)
    cmd = [
        "ffmpeg",
        "-ss",
        f"{start}",
        "-i",
        str(inp),
        "-t",
        f"{dur}",
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
    subprocess.run(cmd, check=True, capture_output=True, text=True)
