from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np

from ..models import Detection


class AnimalDetector(ABC):
    """
    Base interface for animal detectors. Implementations should be used in a
    thread-safe manner or instantiated per worker.
    """

    @abstractmethod
    def warmup(self) -> None:
        """Optional warmup phase (e.g., for GPU)."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> Sequence[Detection]:
        """
        Return a list of `Detection` for a BGR frame (OpenCV convention).
        Coordinates are pixels (x1, y1, x2, y2) relative to the frame.
        """

    def detect_many(self, frames_bgr: Iterable[np.ndarray]) -> Iterable[Sequence[Detection]]:
        """
        Default: call `detect` iteratively. Implementations can override for batch inference.
        """
        for fr in frames_bgr:
            yield self.detect(fr)

    @abstractmethod
    def close(self) -> None:
        """Release resources (model/session)."""
