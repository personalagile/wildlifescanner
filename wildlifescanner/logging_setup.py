from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path


def setup_logging(output_dir: Path, level: str = "INFO") -> Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wildlifescanner")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on repeated setup calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler in the output directory
    fh = logging.FileHandler(output_dir / "wildlifescanner.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
