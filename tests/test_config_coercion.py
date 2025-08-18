from __future__ import annotations

from pathlib import Path

import pytest

from wildlifescanner.config import _coerce_float, _coerce_int, load_config


def test_coerce_float_invalid_returns_default():
    assert _coerce_float("abc", 1.23) == 1.23
    assert _coerce_float(None, 4.56) == 4.56


def test_coerce_int_invalid_returns_default():
    assert _coerce_int("abc", 7) == 7
    assert _coerce_int(None, 9) == 9


def test_load_config_output_dir_mkdir_failure(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    # Create a path to simulate output dir; we'll monkeypatch Path.mkdir to raise
    class DummyError(Exception):
        pass

    called = {"yes": False}

    def fake_mkdir(self, parents=False, exist_ok=False):  # type: ignore[no-untyped-def]
        called["yes"] = True
        raise DummyError("mkdir failed")

    monkeypatch.setattr(Path, "mkdir", fake_mkdir, raising=False)

    # Should not raise despite failing to create output directory
    cfg = load_config(input_dir, None, None)
    assert cfg.input_dir == input_dir
    assert called["yes"] is True
