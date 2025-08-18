from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .pipeline import is_video_file


def wait_until_stable(
    path: Path,
    stable_seconds: float,
    poll_interval: float = 0.5,
    timeout: float = 600.0,
) -> bool:
    """
    Wait until file size and mtime remain stable for `stable_seconds`.
    """
    start_time = time.time()
    last_size = -1
    last_mtime = -1.0
    stable_since = None

    while True:
        try:
            stat = path.stat()
        except FileNotFoundError:
            time.sleep(poll_interval)
            if time.time() - start_time > timeout:
                return False
            continue
        size = stat.st_size
        mtime = stat.st_mtime
        if size == last_size and mtime == last_mtime:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= stable_seconds:
                return True
        else:
            stable_since = None
            last_size = size
            last_mtime = mtime
        if time.time() - start_time > timeout:
            return False
        time.sleep(poll_interval)


class _VideoEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        logger: logging.Logger,
        on_ready: Callable[[Path], None],
        stable_seconds: float,
    ):
        super().__init__()
        self.logger = logger
        self.on_ready = on_ready
        self.stable_seconds = stable_seconds
        self._processing: set[Path] = set()

    def _schedule_process(self, p: Path) -> None:
        if p in self._processing:
            return
        self._processing.add(p)

        def worker():
            try:
                ok = wait_until_stable(p, self.stable_seconds)
                if not ok:
                    self.logger.warning(f"File did not become stable: {p}")
                    return
                self.on_ready(p)
            finally:
                self._processing.discard(p)

        threading.Thread(target=worker, daemon=True).start()

    def on_created(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        p = Path(event.src_path)
        if is_video_file(p):
            self.logger.info(f"New video file detected: {p.name}")
            self._schedule_process(p)

    def on_moved(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        p = Path(event.dest_path)
        if is_video_file(p):
            self.logger.info(f"Video file moved/renamed: {p.name}")
            self._schedule_process(p)


def watch_directory(
    input_dir: Path,
    on_video_ready: Callable[[Path], None],
    stable_seconds: float,
    logger: logging.Logger,
) -> None:
    event_handler = _VideoEventHandler(logger, on_video_ready, stable_seconds)
    observer = Observer()
    observer.schedule(event_handler, str(input_dir), recursive=False)
    observer.start()
    logger.info(f"Watching directory: {input_dir}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopped by KeyboardInterrupt")
    finally:
        observer.stop()
        observer.join()
