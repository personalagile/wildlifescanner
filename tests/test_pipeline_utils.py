from __future__ import annotations

from pathlib import Path

from wildlifescanner.pipeline import is_video_file


def test_is_video_file_basic():
    assert is_video_file(Path("sample.mp4"))
    assert is_video_file(Path("movie.MOV"))
    assert is_video_file(Path("clip.m4v"))
    assert not is_video_file(Path("doc.txt"))
    assert not is_video_file(Path("image.jpeg"))
