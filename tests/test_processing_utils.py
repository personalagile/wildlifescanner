from __future__ import annotations

from pathlib import Path

from wildlifescanner.models import VideoSegment
from wildlifescanner.processing.video import format_segment_filename


def test_format_segment_filename():
    seg = VideoSegment(1.234, 5.678)
    name = format_segment_filename("video123", 2, seg, ".mp4")
    assert name.startswith("video123_seg002_")
    assert name.endswith("ms_5678ms.mp4")
