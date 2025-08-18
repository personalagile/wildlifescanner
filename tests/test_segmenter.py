from __future__ import annotations

from wildlifescanner.segmenter import compute_segments


def test_compute_segments_basic_merge():
    # Ereignisse dicht beisammen, sollten zusammengeführt werden
    times = [1.0, 1.3, 3.8]  # s
    segs = compute_segments(
        activity_times=times,
        video_duration=10.0,
        preroll_sec=0.5,
        postroll_sec=1.0,
        min_activity_sec=0.1,
        merge_gap_sec=0.4,
    )
    # Erwartung: [0.5..2.3] und [3.3..4.8]; der zweite Event erweitert das erste Intervall bis 2.3
    assert len(segs) == 2
    assert round(segs[0].start, 2) == 0.5
    assert round(segs[0].end, 2) == 2.3
    assert round(segs[1].start, 2) == 3.3
    assert round(segs[1].end, 2) == 4.8


def test_compute_segments_min_duration():
    # Sehr kurze Aktivität: sollte herausgefiltert werden
    times = [5.0]
    segs = compute_segments(
        activity_times=times,
        video_duration=6.0,
        preroll_sec=0.0,
        postroll_sec=0.05,
        min_activity_sec=0.2,
        merge_gap_sec=0.1,
    )
    assert len(segs) == 0
