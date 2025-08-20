from __future__ import annotations

import logging
import types
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .cli import parse_args
from .config import load_config
from .detectors.factory import create_detector
from .logging_setup import setup_logging
from .pipeline import analyze_and_extract, is_video_file
from .watcher import watch_directory

if TYPE_CHECKING:
    from .config import AppConfig


def _process_video(path: Path, cfg: AppConfig, logger: logging.Logger) -> None:
    # A/B mode: process with multiple detectors into per-detector subfolders
    if getattr(cfg, "ab_test", False):
        detectors = cfg.ab_detectors or ("YOLO", "MEGADETECTOR")
        logger.info("A/B mode enabled; detectors=%s", ",".join(detectors))
        for det_name in detectors:
            det_upper = det_name.upper()
            if det_upper not in {"YOLO", "MEGADETECTOR"}:
                logger.warning("Skipping unknown detector in A/B list: %s", det_name)
                continue
            # Per-detector output directory
            det_out = cfg.output_dir / f"ab-{det_upper}"
            det_out.mkdir(parents=True, exist_ok=True)
            # Per-detector config clone (works for dataclass AppConfig and SimpleNamespace in tests)
            if is_dataclass(cfg):
                cfg_det = replace(cfg, detector=det_upper, output_dir=det_out)
            else:
                cfg_det = types.SimpleNamespace(**vars(cfg))
                cfg_det.detector = det_upper
                cfg_det.output_dir = det_out
            # Model path resolution per detector
            model_path = cfg.yolo_model
            if det_upper == "MEGADETECTOR" and cfg.megadetector_model:
                model_path = cfg.megadetector_model

            detector = create_detector(
                det_upper,
                yolo_model=model_path,
                confidence=cfg.confidence_threshold,
                iou=cfg.nms_iou,
                allowed_classes=cfg.animal_classes,
            )
            try:
                outputs = analyze_and_extract(path, cfg_det, detector, logger)
                if outputs:
                    logger.info("[%s] Extracted files: %s", det_upper, [p.name for p in outputs])
            finally:
                detector.close()
        return

    # Single-detector mode
    model_path = cfg.yolo_model
    if cfg.detector.upper() == "MEGADETECTOR" and cfg.megadetector_model:
        model_path = cfg.megadetector_model

    detector = create_detector(
        cfg.detector,
        yolo_model=model_path,
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

    # CLI overrides for A/B testing
    if hasattr(args, "ab_test") and args.ab_test:
        cfg.ab_test = True
    if hasattr(args, "ab_detectors") and args.ab_detectors:
        dets = tuple(s.strip().upper() for s in args.ab_detectors.split(",") if s.strip())
        cfg.ab_detectors = dets

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
