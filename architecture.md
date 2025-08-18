# WildlifeScanner — Architecture (arc42)

Version: 0.1

## 1. Introduction and Goals
- Purpose: Monitor an input directory for wildlife camera videos, detect animal activity, and extract relevant segments into an output directory.
- Goals: Modularity (pluggable detectors), reliability (logs, tests), performance (frame stride, FFmpeg copy), maintainability (clean code, typed, tests).

## 2. Constraints
- Python 3.10+.
- FFmpeg binary available on PATH.
- Default detector: Ultralytics YOLOv8. GPU optional.
- Configuration via `.env` in input directory and/or CLI.

## 3. Context and Scope
```
+-----------------+               +--------------------+
| User / Operator |               | Video Sources      |
|  - runs program |               |  - trail cams      |
|  - drops videos |               |  - local files     |
+--------+--------+               +---------+----------+
         |                                  |
         v                                  v
                 +-------------------------------+
                 |         WildlifeScanner       |
                 |  - watcher                    |
                 |  - pipeline (detector+cut)    |
                 |  - logs to output dir         |
                 +-------------------------------+
         ^                                  ^
         |                                  |
+--------+--------+               +---------+----------+
| Output Directory|               | External Tools     |
|  - segments     |               |  - FFmpeg          |
|  - logs         |               |  - YOLO weights    |
+-----------------+               +--------------------+
```

## 4. Solution Strategy
- Separate concerns: detection, segmentation, processing (cutting), orchestration.
- Lazy import heavy deps (cv2, ffmpeg) to keep tests light.
- Stream-copy segments with FFmpeg; fallback to re-encode on failure.
- Provide base detector interface; default YOLO implementation; future MegaDetector support.
- Watch input directory; process when files become stable.

## 5. Building Block View
### Level 1 (Top-level modules)
```
+------------------+   +----------------+   +------------------+
| CLI              |   | Config (.env)  |   | Logging Setup    |
| build/parse args |   | load/merge     |   | console+file     |
+---------+--------+   +--------+-------+   +---------+--------+
          |                         \                /
          v                          v              v
                 +-----------------------------+
                 |           Main              |
                 | parse args -> load config   |
                 | setup logging               |
                 | start watcher               |
                 +--------------+--------------+
                                |
                                v
                        +---------------+
                        |   Watcher     |
                        |  (watchdog)   |
                        +-------+-------+
                                |
                                v
                         +-------------+
                         |  Pipeline   |
                         +------+------+
                                |
        +-----------------------+-----------------------+
        v                                               v
+---------------+                             +------------------+
| Detectors     |                             | Processing/Video |
|  - base       |                             |  - probe/cut     |
|  - YOLO       |                             +------------------+
+-------+-------+                                     ^
        |                                             |
        v                                             |
   +-----------+                                      |
   | Segmenter |  (build segments from activity)      |
   +-----------+--------------------------------------+
```

### Level 2 (Key responsibilities)
- `wildlifescanner/cli.py`: `build_parser()`, `parse_args()`.
- `wildlifescanner/config.py`: `AppConfig`, `load_config()` preferring CLI over `.env` over defaults.
- `wildlifescanner/logging_setup.py`: console and file handlers.
- `wildlifescanner/watcher.py`: `watch_directory()`, `wait_until_stable()`.
- `wildlifescanner/pipeline.py`: `is_video_file()`, `analyze_and_extract()` coordinates the flow.
- `wildlifescanner/detectors/base.py`: `AnimalDetector` interface.
- `wildlifescanner/detectors/yolo.py`: `YOLODetector` using Ultralytics.
- `wildlifescanner/processing/video.py`: `probe_video()`, `extract_segments()`.
- `wildlifescanner/segmenter.py`: `compute_segments()`.

## 6. Runtime View (Typical processing)
```
User drops video -> Watcher detects -> wait_until_stable
   -> Pipeline.analyze_and_extract
      -> probe_video (fps, frames, duration)
      -> for frames with stride: detector.detect(frame)
         -> if any detections: record activity time
      -> compute_segments(activity, preroll, postroll, merge, min)
      -> extract_segments via FFmpeg (copy -> re-encode fallback)
      -> write outputs + log results
```

## 7. Deployment View
```
+------------------------------+
| Host: local machine          |
| - Python 3.10+ venv          |
| - FFmpeg installed           |
| - Optional: GPU/CUDA         |
+------------------------------+
```
Single-process Python app, watching a local directory.

## 8. Cross-cutting Concepts
- Configuration: `.env` loaded from input directory; CLI can override.
- Logging: structured format to console and `wildlifescanner.log` in output dir.
- Error handling: cautious with IO; falls back on re-encode; watcher waits for stable files.
- Performance: frame stride; YOLO confidence/IoU; stream-copy to avoid re-encode.
- Testability: lazy imports; dummy/unit tests; optional video-set integration tests behind env flag.

## 9. Architectural Decisions (ADRs)
- ADR-001: Use YOLOv8 (Ultralytics) as default detector.
- ADR-002: FFmpeg for cutting (copy first, fallback to re-encode).
- ADR-003: Watchdog for file system watching.
- ADR-004: `.env` in input dir for operational config.

## 10. Quality Requirements
- Reliability: skip unstable files; handle detector init errors; clear logs.
- Maintainability: modular packages, type hints, tests, linters.
- Performance: process with stride; stream-copy; optional GPU.
- Testability: unit tests; integration video-set tests opt-in via `RUN_VIDEO_TESTS`.

## 11. Risks and Technical Debt
- Detector model weights availability/network constraints.
- FFmpeg incompatibilities on some platforms.
- Class coverage limited by default COCO model.
- Future: MegaDetector integration, tracking/cropping, batching.

## 12. Glossary
- Activity: at least one detection in a frame.
- Segment: time interval [start, end] extracted into a new video file.
- Pre-roll/Post-roll: padding added around detected activity.
