from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


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
    end: float    # seconds

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


Detections = Sequence[Detection]
