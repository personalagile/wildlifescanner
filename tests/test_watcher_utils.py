from __future__ import annotations

from pathlib import Path
import time

from wildlifescanner.watcher import wait_until_stable


def test_wait_until_stable_true(tmp_path: Path):
    f = tmp_path / "file.bin"
    f.write_bytes(b"123")
    assert wait_until_stable(f, stable_seconds=0.1, poll_interval=0.02, timeout=1.0)


def test_wait_until_stable_timeout(tmp_path: Path):
    # Non-existent path should time out quickly
    missing = tmp_path / "missing.txt"
    assert not wait_until_stable(missing, stable_seconds=0.1, poll_interval=0.02, timeout=0.3)
