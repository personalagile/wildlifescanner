from __future__ import annotations

from pathlib import Path

from wildlifescanner.config import AppConfig, load_config


def test_load_config_defaults(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    cfg = load_config(input_dir, None, None)
    assert isinstance(cfg, AppConfig)
    assert cfg.input_dir == input_dir
    assert cfg.output_dir.exists()
    assert cfg.detector in {"YOLO", "MEGADETECTOR"}


def test_load_config_env_override(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    env = input_dir / ".env"
    env.write_text(
        """
OUTPUT_DIR={out}
DETECTOR=YOLO
YOLO_MODEL=yolov8n.pt
PREROLL_SEC=2.5
MIN_ACTIVITY_SEC=0.7
ANIMAL_CLASSES=fox, deer
        """.strip().format(out=tmp_path / "out")
    )

    cfg = load_config(input_dir, None, None)
    assert cfg.output_dir == (tmp_path / "out").resolve()
    assert cfg.preroll_sec == 2.5
    assert cfg.min_activity_sec == 0.7
    assert cfg.animal_classes == ("fox", "deer")
