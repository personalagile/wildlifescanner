from __future__ import annotations

from pathlib import Path

from wildlifescanner.config import AppConfig, load_config


def test_load_config_ab_env(tmp_path: Path):
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / ".env").write_text(
        """
AB_TEST=true
AB_DETECTORS= yolo , MegaDetector
        """.strip()
    )

    cfg = load_config(inp, None, None)
    assert isinstance(cfg, AppConfig)
    assert cfg.ab_test is True
    # Normalized and uppercased, order preserved
    assert cfg.ab_detectors == ("YOLO", "MEGADETECTOR")


def test_load_config_ab_default_when_missing(tmp_path: Path):
    inp = tmp_path / "input"
    inp.mkdir()
    # No AB_DETECTORS provided; AB_TEST false by default
    cfg = load_config(inp, None, None)
    assert cfg.ab_test is False
    assert cfg.ab_detectors == ()
