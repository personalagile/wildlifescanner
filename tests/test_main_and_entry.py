from __future__ import annotations

import runpy
import types
from pathlib import Path

import pytest

import wildlifescanner.main as main_mod
from wildlifescanner.config import AppConfig


def test_main_invokes_watch_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Prepare config with temp dirs
    inp = tmp_path / "in"
    out = tmp_path / "out"

    # parse_args returns placeholder values; load_config will supply actual cfg
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda argv=None: types.SimpleNamespace(
            input=inp,
            output=out,
            detector=None,
        ),
    )

    cfg = AppConfig(input_dir=inp, output_dir=out)

    # Stub load_config to return our cfg
    monkeypatch.setattr(main_mod, "load_config", lambda a, b, c: cfg)

    # Stub logger with info method
    logs: list[str] = []

    class _Log:
        def info(self, msg: str):
            logs.append(str(msg))

    monkeypatch.setattr(main_mod, "setup_logging", lambda out_dir, level=None: _Log())

    # Create a pre-existing video file to hit listing path
    inp.mkdir(parents=True, exist_ok=True)
    (inp / "preexisting.mp4").write_bytes(b"")

    called: dict[str, object] = {}

    def _watch(input_dir: Path, on_video_ready, stable_seconds: float, logger):
        called["input"] = input_dir
        called["stable"] = stable_seconds
        called["cb"] = on_video_ready
        called["logger"] = logger
        # do not loop

    monkeypatch.setattr(main_mod, "watch_directory", _watch)

    # Run
    main_mod.main([])

    # Assertions
    assert called["input"] == inp
    assert called["stable"] == cfg.file_stability_seconds
    assert inp.exists() and out.exists()
    # Logged intro lines and found pre-existing
    assert any("WildlifeScanner started" in s for s in logs)
    assert any("Found existing file:" in s for s in logs)


def test___main___entrypoint_calls_main(monkeypatch: pytest.MonkeyPatch):
    called = {"ok": False}

    def _stub_main(argv=None):
        called["ok"] = True

    # Ensure the package main is stubbed before executing module as __main__
    import wildlifescanner.main as pkg_main

    monkeypatch.setattr(pkg_main, "main", _stub_main)

    # Re-run the __main__ module under __main__ semantics
    runpy.run_module("wildlifescanner.__main__", run_name="__main__")

    assert called["ok"] is True
