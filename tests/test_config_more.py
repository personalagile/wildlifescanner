from __future__ import annotations

from pathlib import Path

from wildlifescanner.config import _coerce_bool, load_config


def test_coerce_bool_unknown_returns_default():
    assert _coerce_bool("maybe", True) is True
    assert _coerce_bool("unknown", False) is False


def test_load_config_cli_output_overrides(tmp_path: Path):
    inp = tmp_path / "in"
    out = tmp_path / "custom_out"
    inp.mkdir()
    cfg = load_config(cli_input=inp, cli_output=out, cli_detector=None)
    assert cfg.output_dir == out
    assert cfg.input_dir == inp
    # load_config should ensure directory exists (suppressed exceptions aside)
    assert out.exists()
