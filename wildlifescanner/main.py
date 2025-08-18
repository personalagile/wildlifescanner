from __future__ import annotations

import logging
from pathlib import Path

from .cli import parse_args
from .config import AppConfig, load_config
from .detectors.factory import create_detector
from .logging_setup import setup_logging
from .pipeline import analyze_and_extract, is_video_file
from .watcher import watch_directory


def _process_video(path: Path, cfg: AppConfig, logger: logging.Logger) -> None:
    detector = create_detector(
        cfg.detector,
        yolo_model=cfg.yolo_model,
        confidence=cfg.confidence_threshold,
        iou=cfg.nms_iou,
        allowed_classes=cfg.animal_classes,
    )
    try:
        outputs = analyze_and_extract(path, cfg, detector, logger)
        if outputs:
            logger.info(f"Extracted files: {[p.name for p in outputs]}")
    finally:
        detector.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.input, args.output, args.detector)

    logger = setup_logging(cfg.output_dir, level=cfg.log_level)
    logger.info("WildlifeScanner started")
    logger.info(f"Input: {cfg.input_dir}")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info(f"Detector: {cfg.detector}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.input_dir.mkdir(parents=True, exist_ok=True)

    # Optionally list pre-existing videos (for now, log only)
    for p in sorted(cfg.input_dir.iterdir()):
        if p.is_file() and is_video_file(p):
            logger.info(f"Found existing file: {p.name}")

    def on_video_ready(p: Path) -> None:
        try:
            _process_video(p, cfg, logger)
        except Exception as e:
            logger.exception(f"Error while processing {p}: {e}")

    watch_directory(cfg.input_dir, on_video_ready, cfg.file_stability_seconds, logger)
