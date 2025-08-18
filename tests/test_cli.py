from __future__ import annotations

from wildlifescanner.cli import parse_args


def test_parse_args_defaults():
    ns = parse_args([])
    assert hasattr(ns, "input")
    assert hasattr(ns, "output")
    assert ns.detector is None


def test_parse_args_detector_choice():
    ns = parse_args(["--detector", "YOLO"])
    assert ns.detector == "YOLO"
