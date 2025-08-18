from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls_name: str


@dataclass(frozen=True)
class VideoSegment:
    start: float  # seconds
    end: float  # seconds

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


Detections = Sequence[Detection]
