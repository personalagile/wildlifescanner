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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
