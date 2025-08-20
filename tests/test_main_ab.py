from __future__ import annotations

import types
from pathlib import Path

import pytest

import wildlifescanner.main as main_mod


def test__process_video_ab_writes_to_per_detector_subfolders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    created: list[tuple[str, str]] = []
    calls: list[tuple[str, Path]] = []

    class _Det:
        def __init__(self, name: str):
            self.name = name

        def close(self) -> None:  # pragma: no cover - trivial
            pass

    def fake_create(
        det_name: str,
        *,
        yolo_model: str,
        confidence: float,
        iou: float,
        allowed_classes: tuple[str, ...],
    ):  # type: ignore[no-untyped-def]
        created.append((det_name, yolo_model))
        return _Det(det_name)

    def fake_analyze(_path: Path, cfg, detector: _Det, _logger):  # type: ignore[no-untyped-def]
        # Output should be routed into per-detector subfolder
        assert cfg.output_dir.name == f"ab-{detector.name}"
        out_file = cfg.output_dir / "seg001.mp4"
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"")
        calls.append((detector.name, cfg.output_dir))
        return [out_file]

    monkeypatch.setattr(main_mod, "create_detector", fake_create)
    monkeypatch.setattr(main_mod, "analyze_and_extract", fake_analyze)

    cfg = types.SimpleNamespace(
        ab_test=True,
        ab_detectors=("YOLO", "MEGADETECTOR"),
        detector="YOLO",
        yolo_model="m.pt",
        megadetector_model="md.pt",
        confidence_threshold=0.2,
        nms_iou=0.5,
        animal_classes=("animal",),
        output_dir=tmp_path / "out",
        input_dir=tmp_path / "in",
        log_level="INFO",
    )

    video = tmp_path / "video.mp4"
    video.write_bytes(b"")

    main_mod._process_video(video, cfg, logger=types.SimpleNamespace(info=lambda *a, **k: None))

    # Both detectors created with their respective model paths
    assert ("YOLO", "m.pt") in created
    assert ("MEGADETECTOR", "md.pt") in created

    # Both analyze calls happened and outputs are in the right subfolders
    det_names = [n for n, _ in calls]
    assert det_names == ["YOLO", "MEGADETECTOR"]
    assert (cfg.output_dir / "ab-YOLO" / "seg001.mp4").exists()
    assert (cfg.output_dir / "ab-MEGADETECTOR" / "seg001.mp4").exists()


def test_main_cli_ab_overrides_config_and_triggers_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    inp = tmp_path / "in"
    out = tmp_path / "out"

    # parse_args returns AB flags set via CLI
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda argv=None: types.SimpleNamespace(
            input=inp,
            output=out,
            detector=None,
            ab_test=True,
            ab_detectors="YOLO,MEGADETECTOR",
        ),
    )

    # load_config returns baseline cfg with AB flags off
    base_cfg = types.SimpleNamespace(
        input_dir=inp,
        output_dir=out,
        detector="YOLO",
        yolo_model="m.pt",
        megadetector_model="md.pt",
        confidence_threshold=0.25,
        nms_iou=0.45,
        animal_classes=("animal",),
        file_stability_seconds=0.1,
        log_level="INFO",
        ab_test=False,
        ab_detectors=(),
    )
    monkeypatch.setattr(main_mod, "load_config", lambda a, b, c: base_cfg)

    # Stub logging
    monkeypatch.setattr(
        main_mod,
        "setup_logging",
        lambda *_a, **_k: types.SimpleNamespace(info=lambda *a, **k: None),
    )

    # When main installs the callback, run it once to exercise _process_video with overridden cfg
    def fake_watch(_input_dir: Path, on_video_ready, _stable: float, _logger):  # type: ignore[no-untyped-def]
        # Ensure overrides were applied on cfg captured by closure inside main
        # We can only assert by calling the callback and having _process_video check cfg
        on_video_ready(tmp_path / "v.mp4")

    monkeypatch.setattr(main_mod, "watch_directory", fake_watch)

    # Intercept _process_video to assert AB flags propagated
    called = {"ok": False}

    def fake_proc(_p: Path, cfg, _logger):  # type: ignore[no-untyped-def]
        assert cfg.ab_test is True
        assert cfg.ab_detectors == ("YOLO", "MEGADETECTOR")
        called["ok"] = True

    monkeypatch.setattr(main_mod, "_process_video", fake_proc)

    main_mod.main([])
    assert called["ok"] is True
