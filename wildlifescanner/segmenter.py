from __future__ import annotations

from typing import Iterable, List, Sequence

from .models import VideoSegment


def compute_segments(
    activity_times: Sequence[float],
    video_duration: float,
    preroll_sec: float,
    postroll_sec: float,
    min_activity_sec: float,
    merge_gap_sec: float,
) -> List[VideoSegment]:
    """
    Build segments from timestamps where activity was detected.

    Strategy: Expand each event to [t - preroll, t + postroll], then merge
    overlapping/adjacent intervals (with merge gap), and drop segments
    below the minimum duration.
    """
    if not activity_times:
        return []

    times = sorted(float(t) for t in activity_times if t >= 0.0)
    intervals: List[VideoSegment] = []
    for t in times:
        start = max(0.0, t - preroll_sec)
        end = min(video_duration, t + postroll_sec)
        if end > start:
            intervals.append(VideoSegment(start, end))

    if not intervals:
        return []

    merged: List[VideoSegment] = []
    cur = intervals[0]
    for seg in intervals[1:]:
        if seg.start <= cur.end + merge_gap_sec:
            # merge
            cur = VideoSegment(cur.start, max(cur.end, seg.end))
        else:
            if cur.duration() >= min_activity_sec:
                merged.append(cur)
            cur = seg
    if cur.duration() >= min_activity_sec:
        merged.append(cur)

    return merged
