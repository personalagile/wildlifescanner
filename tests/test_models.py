from __future__ import annotations

from wildlifescanner.models import VideoSegment


def test_video_segment_duration_positive():
    seg = VideoSegment(1.0, 3.5)
    assert seg.duration() == 2.5


def test_video_segment_duration_non_negative():
    seg = VideoSegment(5.0, 2.0)
    assert seg.duration() == 0.0
