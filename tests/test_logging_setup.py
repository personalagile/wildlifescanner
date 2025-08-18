from __future__ import annotations

import logging
from pathlib import Path

from wildlifescanner.logging_setup import setup_logging


def test_setup_logging_creates_file_and_handlers(tmp_path: Path):
    logdir = tmp_path / "logs"
    logger = setup_logging(logdir, level="DEBUG")
    # Level set
    assert logger.level == logging.DEBUG
    # Handlers present (console + file)
    assert len(logger.handlers) >= 2
    # Log once and check file exists
    logger.debug("test message")
    logfile = logdir / "wildlifescanner.log"
    assert logfile.exists()
    # Cleanup handlers to avoid cross-test interference
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
