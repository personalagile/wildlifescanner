from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wildlifescanner",
        description="Extracts animal-activity segments from wildlife camera videos",
    )
    parser.add_argument("--input", type=Path, required=False, help="Input directory")
    parser.add_argument("--output", type=Path, required=False, help="Output directory")
    parser.add_argument(
        "--detector",
        type=str,
        choices=["YOLO", "MEGADETECTOR"],
        help="Detector selection (overrides .env)",
    )
    parser.add_argument(
        "--ab-test",
        action="store_true",
        help="Enable A/B mode and process each video with multiple detectors",
    )
    parser.add_argument(
        "--ab-detectors",
        type=str,
        help='Comma-separated list of detectors for A/B mode, e.g. "YOLO,MEGADETECTOR"',
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
