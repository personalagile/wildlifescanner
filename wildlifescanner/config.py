from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

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

    # Tracking/Zooming options
    zoom_enabled: bool = False
    tracking_enabled: bool = False
    # Minimal output resolution for zoomed videos (width x height)
    min_output_width: int = 640
    min_output_height: int = 360
    # If True, keep post-processed file with suffix instead of replacing original
    keep_postprocessed: bool = False

    # MegaDetector (PyTorch) model path
    megadetector_model: str = ""

    # A/B testing
    ab_test: bool = False
    ab_detectors: tuple[str, ...] = ()

    # Tracking smoothing parameters (tunable)
    tracking_center_alpha: float = 0.05
    tracking_size_alpha: float = 0.04
    tracking_max_move_frac: float = 0.05
    tracking_max_zoom_frac: float = 0.06
    tracking_center_deadzone_frac: float = 0.10
    tracking_zoom_deadzone_frac: float = 0.12
    tracking_margin: float = 0.20


def _get_env_from_file(env_path: Path) -> dict[str, str]:
    if env_path.exists():
        return {k: v for k, v in dotenv_values(env_path).items() if k and v}
    return {}


def _coerce_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _coerce_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def load_config(
    cli_input: Path | None,
    cli_output: Path | None,
    cli_detector: str | None,
) -> AppConfig:
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
    output_dir: Path
    if cli_output is not None:
        output_dir = cli_output
    else:
        env_out = env_from_input.get("OUTPUT_DIR")
        if env_out is not None:
            output_dir = Path(env_out)
        else:
            default_output = input_dir.parent / "output"
            output_dir = Path(os.getenv("OUTPUT_DIR", str(default_output))).resolve()

    # 3) Detector: CLI > .env > default
    detector = cli_detector or env_from_input.get("DETECTOR", "YOLO").upper()

    # 4) Additional parameters
    yolo_model = env_from_input.get("YOLO_MODEL", "yolov8n.pt")
    megadetector_model = env_from_input.get("MEGADETECTOR_MODEL", "")
    # A/B testing
    ab_test = _coerce_bool(env_from_input.get("AB_TEST"), False)
    ab_detectors_env = env_from_input.get("AB_DETECTORS", "")
    ab_detectors = tuple(s.strip().upper() for s in ab_detectors_env.split(",") if s.strip())
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

    # 5) Tracking/Zooming
    zoom_enabled = _coerce_bool(env_from_input.get("ZOOM_ENABLED"), False)
    tracking_enabled = _coerce_bool(env_from_input.get("TRACKING_ENABLED"), False)
    min_output_width = _coerce_int(env_from_input.get("MIN_OUTPUT_WIDTH"), 640)
    min_output_height = _coerce_int(env_from_input.get("MIN_OUTPUT_HEIGHT"), 360)
    keep_postprocessed = _coerce_bool(env_from_input.get("KEEP_POSTPROCESSED"), False)

    tracking_center_alpha = _coerce_float(env_from_input.get("TRACKING_CENTER_ALPHA"), 0.05)
    tracking_size_alpha = _coerce_float(env_from_input.get("TRACKING_SIZE_ALPHA"), 0.04)
    tracking_max_move_frac = _coerce_float(env_from_input.get("TRACKING_MAX_MOVE_FRAC"), 0.05)
    tracking_max_zoom_frac = _coerce_float(env_from_input.get("TRACKING_MAX_ZOOM_FRAC"), 0.06)
    tracking_center_deadzone_frac = _coerce_float(
        env_from_input.get("TRACKING_CENTER_DEADZONE_FRAC"), 0.10
    )
    tracking_zoom_deadzone_frac = _coerce_float(
        env_from_input.get("TRACKING_ZOOM_DEADZONE_FRAC"), 0.12
    )
    tracking_margin = _coerce_float(env_from_input.get("TRACKING_MARGIN"), 0.20)

    # Ensure the output directory exists (test expectation)
    with contextlib.suppress(Exception):
        output_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        detector=detector,
        yolo_model=yolo_model,
        megadetector_model=megadetector_model,
        ab_test=ab_test,
        ab_detectors=ab_detectors,
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
        zoom_enabled=zoom_enabled,
        tracking_enabled=tracking_enabled,
        min_output_width=min_output_width,
        min_output_height=min_output_height,
        tracking_center_alpha=tracking_center_alpha,
        tracking_size_alpha=tracking_size_alpha,
        tracking_max_move_frac=tracking_max_move_frac,
        tracking_max_zoom_frac=tracking_max_zoom_frac,
        tracking_center_deadzone_frac=tracking_center_deadzone_frac,
        tracking_zoom_deadzone_frac=tracking_zoom_deadzone_frac,
        tracking_margin=tracking_margin,
        keep_postprocessed=keep_postprocessed,
    )
