from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


@dataclass
class AppConfig:
    input_dir: Path
    output_dir: Path

    detector: str = "YOLO"  # YOLO | MEGADETECTOR
    yolo_model: str = "yolov8n.pt"
    animal_classes: tuple[str, ...] = (
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    )

    confidence_threshold: float = 0.25
    nms_iou: float = 0.45
    frame_stride: int = 5

    preroll_sec: float = 1.0
    postroll_sec: float = 2.0
    min_activity_sec: float = 0.5
    merge_gap_sec: float = 1.0

    file_stability_seconds: float = 3.0
    poll_interval_seconds: float = 1.0

    log_level: str = "INFO"


def _get_env_from_file(env_path: Path) -> dict[str, str]:
    if env_path.exists():
        return {k: v for k, v in dotenv_values(env_path).items() if k and v}
    return {}


def _coerce_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _coerce_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def load_config(cli_input: Optional[Path], cli_output: Optional[Path], cli_detector: Optional[str]) -> AppConfig:
    """
    Load configuration with the following priority order:
    1) CLI arguments (input/output/detector)
    2) .env in the input directory
    3) Defaults
    """

    # 1) Base: input/output from CLI or environment
    input_dir = cli_input or Path(os.getenv("INPUT_DIR", ".")).resolve()
    env_from_input = _get_env_from_file(input_dir / ".env")

    # 2) Output: CLI > .env > default: ./output next to input
    output_dir = (
        cli_output
        or Path(env_from_input.get("OUTPUT_DIR")) if env_from_input.get("OUTPUT_DIR") else None
    )
    if output_dir is None:
        default_output = input_dir.parent / "output"
        output_dir = Path(os.getenv("OUTPUT_DIR", str(default_output))).resolve()

    # 3) Detector: CLI > .env > default
    detector = cli_detector or env_from_input.get("DETECTOR", "YOLO").upper()

    # 4) Additional parameters
    yolo_model = env_from_input.get("YOLO_MODEL", "yolov8n.pt")
    animal_classes = tuple(
        s.strip() for s in env_from_input.get("ANIMAL_CLASSES", "").split(",") if s.strip()
    ) or (
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    )

    confidence_threshold = _coerce_float(env_from_input.get("CONFIDENCE_THRESHOLD"), 0.25)
    nms_iou = _coerce_float(env_from_input.get("NMS_IOU"), 0.45)
    frame_stride = _coerce_int(env_from_input.get("FRAME_STRIDE"), 5)

    preroll_sec = _coerce_float(env_from_input.get("PREROLL_SEC"), 1.0)
    postroll_sec = _coerce_float(env_from_input.get("POSTROLL_SEC"), 2.0)
    min_activity_sec = _coerce_float(env_from_input.get("MIN_ACTIVITY_SEC"), 0.5)
    merge_gap_sec = _coerce_float(env_from_input.get("MERGE_GAP_SEC"), 1.0)

    file_stability_seconds = _coerce_float(env_from_input.get("FILE_STABILITY_SECONDS"), 3.0)
    poll_interval_seconds = _coerce_float(env_from_input.get("POLL_INTERVAL_SECONDS"), 1.0)

    log_level = env_from_input.get("LOG_LEVEL", "INFO").upper()

    # Ensure the output directory exists (test expectation)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If creation fails, it will be logged later on startup
        pass

    return AppConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        detector=detector,
        yolo_model=yolo_model,
        animal_classes=animal_classes,
        confidence_threshold=confidence_threshold,
        nms_iou=nms_iou,
        frame_stride=frame_stride,
        preroll_sec=preroll_sec,
        postroll_sec=postroll_sec,
        min_activity_sec=min_activity_sec,
        merge_gap_sec=merge_gap_sec,
        file_stability_seconds=file_stability_seconds,
        poll_interval_seconds=poll_interval_seconds,
        log_level=log_level,
    )
