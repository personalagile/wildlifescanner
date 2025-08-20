from __future__ import annotations

from wildlifescanner.cli import parse_args


def test_parse_args_ab_flags_defaults():
    ns = parse_args([])
    assert hasattr(ns, "ab_test")
    assert hasattr(ns, "ab_detectors")
    assert ns.ab_test is False
    assert ns.ab_detectors is None


def test_parse_args_ab_flags_provided():
    ns = parse_args(["--ab-test", "--ab-detectors", "YOLO,MEGADETECTOR"])
    assert ns.ab_test is True
    # ab_detectors is a raw string; main() splits/uppercases
    assert ns.ab_detectors == "YOLO,MEGADETECTOR"
