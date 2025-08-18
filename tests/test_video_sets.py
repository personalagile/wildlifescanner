from __future__ import annotations

import os
from pathlib import Path

import pytest

from wildlifescanner.config import AppConfig
from wildlifescanner.detectors.factory import create_detector
from wildlifescanner.logging_setup import setup_logging
from wildlifescanner.pipeline import analyze_and_extract, is_video_file

RUN_VIDEO_TESTS = os.getenv("RUN_VIDEO_TESTS", "0") == "1"
VIDEOS_ROOT = Path(__file__).parent / "data"
POS_DIR = VIDEOS_ROOT / "positive"
NEG_DIR = VIDEOS_ROOT / "negative"


def _collect_videos(p: Path) -> list[Path]:
    return [f for f in sorted(p.iterdir()) if f.is_file() and is_video_file(f)]


@pytest.mark.integration
@pytest.mark.video_sets
@pytest.mark.skipif(not RUN_VIDEO_TESTS, reason="Set RUN_VIDEO_TESTS=1 to enable video set tests")
def test_positive_videos_have_detections(tmp_path: Path):
    vids = _collect_videos(POS_DIR)
    if not vids:
        pytest.skip("No positive videos found in tests/data/positive")

    # Config: output to tmp, default detector (YOLO)
    cfg = AppConfig(input_dir=tmp_path, output_dir=tmp_path / "out")
    logger = setup_logging(cfg.output_dir, level="INFO")

    # Create detector; skip if model cannot be initialized (e.g., missing weights/network)
    try:
        det = create_detector(
            cfg.detector,
            yolo_model=cfg.yolo_model,
            confidence=cfg.confidence_threshold,
            iou=cfg.nms_iou,
            allowed_classes=cfg.animal_classes,
        )
    except Exception as e:  # pragma: no cover - only in CI environments without models
        pytest.skip(f"Detector init failed: {e}")

    try:
        for v in vids:
            outs = analyze_and_extract(v, cfg, det, logger)
            assert isinstance(outs, list)
            assert len(outs) > 0, f"Expected detections/segments for {v.name}"
    finally:
        det.close()
        # detach handlers to avoid cross-test interference
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()


@pytest.mark.integration
@pytest.mark.video_sets
@pytest.mark.skipif(not RUN_VIDEO_TESTS, reason="Set RUN_VIDEO_TESTS=1 to enable video set tests")
def test_negative_videos_have_no_detections(tmp_path: Path):
    vids = _collect_videos(NEG_DIR)
    if not vids:
        pytest.skip("No negative videos found in tests/data/negative")

    cfg = AppConfig(input_dir=tmp_path, output_dir=tmp_path / "out")
    logger = setup_logging(cfg.output_dir, level="INFO")

    try:
        det = create_detector(
            cfg.detector,
            yolo_model=cfg.yolo_model,
            confidence=cfg.confidence_threshold,
            iou=cfg.nms_iou,
            allowed_classes=cfg.animal_classes,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Detector init failed: {e}")

    try:
        for v in vids:
            outs = analyze_and_extract(v, cfg, det, logger)
            assert isinstance(outs, list)
            assert len(outs) == 0, f"Expected no detections/segments for {v.name}"
    finally:
        det.close()
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
