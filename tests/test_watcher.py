from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

import wildlifescanner.watcher as watcher


def test_event_handler_on_created_and_moved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Make thread run synchronously
    class _T:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            if self.target:
                self.target()

    monkeypatch.setattr("wildlifescanner.watcher.threading.Thread", _T)
    monkeypatch.setattr("wildlifescanner.watcher.wait_until_stable", lambda p, s: True)

    hits: list[Path] = []
    logger = logging.getLogger("test")
    h = watcher._VideoEventHandler(logger, hits.append, stable_seconds=0.0)

    p1 = tmp_path / "a.mp4"
    p2 = tmp_path / "b.mkv"

    # created path
    ev1 = types.SimpleNamespace(is_directory=False, src_path=str(p1))
    h.on_created(ev1)
    # moved path
    ev2 = types.SimpleNamespace(is_directory=False, dest_path=str(p2))
    h.on_moved(ev2)

    assert hits == [p1, p2]


def test_event_handler_dedup_scheduling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Capture threads without running them automatically
    threads: list[object] = []

    class _T:
        def __init__(self, target=None, daemon=None):
            self.target = target
            threads.append(self)

        def start(self):
            # do not run yet
            pass

    monkeypatch.setattr("wildlifescanner.watcher.threading.Thread", _T)
    monkeypatch.setattr("wildlifescanner.watcher.wait_until_stable", lambda p, s: True)

    hits: list[Path] = []
    logger = logging.getLogger("test")
    h = watcher._VideoEventHandler(logger, hits.append, stable_seconds=0.0)

    p = tmp_path / "dup.mp4"
    h._schedule_process(p)
    h._schedule_process(p)  # should be ignored while first is pending

    assert len(threads) == 1  # only one worker scheduled

    # Now run the worker and ensure callback fires once
    threads[0].target()
    assert hits == [p]


def test_watch_directory_keyboardinterrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    created: dict[str, object] = {}

    class _Obs:
        def __init__(self):
            created["inst"] = self
            self.scheduled = []
            self.started = False
            self.stopped = False
            self.joined = False

        def schedule(self, handler, path, recursive=False):
            self.scheduled.append((handler, path, recursive))

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self):
            self.joined = True

    # Inject observer stub
    monkeypatch.setattr("wildlifescanner.watcher.Observer", _Obs)

    # Make the loop raise KeyboardInterrupt on first sleep
    def _sleep(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr("wildlifescanner.watcher.time.sleep", _sleep)

    called: list[Path] = []
    logger = logging.getLogger("test")
    inp = tmp_path

    watcher.watch_directory(inp, called.append, stable_seconds=0.1, logger=logger)

    obs: _Obs = created["inst"]  # type: ignore[assignment]
    assert obs.started is True
    assert obs.stopped is True
    assert obs.joined is True
    assert len(obs.scheduled) == 1
    # scheduled path is string, compare to str(inp)
    assert obs.scheduled[0][1] == str(inp)
    assert obs.scheduled[0][2] is False
