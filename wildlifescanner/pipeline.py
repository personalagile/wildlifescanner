from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from .config import AppConfig
from .detectors.base import AnimalDetector
from .models import VideoSegment
from .processing.video import (
    extract_segments,
    postprocess_zoom_and_tracking,
    probe_video,
)
from .segmenter import compute_segments

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def analyze_and_extract(
    video_path: Path,
    cfg: AppConfig,
    detector: AnimalDetector,
    logger: logging.Logger,
) -> list[Path]:
    # lazy import cv2 in functions
    import cv2  # type: ignore

    fps, frame_count, duration = probe_video(video_path)
    logger.info(
        "Analyzing: %s | fps=%.2f frames=%d duration=%.2fs",
        video_path.name,
        fps,
        frame_count,
        duration,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    activity_times: list[float] = []
    frame_idx = 0
    stride = max(1, int(cfg.frame_stride))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            detections = detector.detect(frame)
            if detections:
                t = frame_idx / fps
                activity_times.append(t)
            frame_idx += 1
    finally:
        cap.release()

    segments: list[VideoSegment] = compute_segments(
        activity_times,
        video_duration=duration,
        preroll_sec=cfg.preroll_sec,
        postroll_sec=cfg.postroll_sec,
        min_activity_sec=cfg.min_activity_sec,
        merge_gap_sec=cfg.merge_gap_sec,
    )

    if not segments:
        logger.info("No activity detected — no segments extracted.")
        return []

    logger.info(f"Detected segments: {[(round(s.start, 2), round(s.end, 2)) for s in segments]}")

    outputs = extract_segments(video_path, cfg.output_dir, segments, logger)

    # Optional zoom/tracking post-process (disabled by default)
    if cfg.zoom_enabled or cfg.tracking_enabled:
        logger.info(
            "Post-processing enabled (tracking=%s, zoom=%s) for %d segment(s)",
            cfg.tracking_enabled,
            cfg.zoom_enabled,
            len(outputs),
        )
        processed_paths: list[Path] = []
        keep_post = getattr(cfg, "keep_postprocessed", False)
        for p in outputs:
            tmp = p.with_name(p.stem + ("_track" if cfg.tracking_enabled else "_zoom") + p.suffix)
            logger.info("Post-processing segment: %s -> %s", p.name, tmp.name)
            changed = False
            try:
                changed = postprocess_zoom_and_tracking(
                    input_video=p,
                    output_video=tmp,
                    cfg=cfg,
                    detector=detector,
                    logger=logger,
                )
                if changed:
                    if keep_post:
                        # Keep processed file alongside original
                        logger.info(
                            "Keeping post-processed file alongside original: %s",
                            tmp.name,
                        )
                    else:
                        # Replace original with processed file
                        tmp.replace(p)
                        logger.info(
                            "Replaced original with post-processed result: %s",
                            p.name,
                        )
                else:
                    logger.info(
                        "No post-processing applied or no detections for: %s (original kept)",
                        p.name,
                    )
            except Exception as e:
                logger.exception("Post-processing failed for %s: %s", p.name, e)
            finally:
                # Clean up tmp if it still exists (e.g., on failure or already moved)
                if tmp.exists():
                    # If keeping processed, do not delete
                    if keep_post and changed:
                        pass
                    else:
                        with contextlib.suppress(Exception):
                            tmp.unlink()
                            logger.info("Cleaned up temporary file: %s", tmp.name)
            processed_paths.append(p)
        return processed_paths

    return outputs
