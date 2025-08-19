# WildlifeScanner

[![CI](https://github.com/personalagile/wildlifescanner/actions/workflows/ci.yml/badge.svg)](https://github.com/personalagile/wildlifescanner/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/personalagile/wildlifescanner/branch/main/graph/badge.svg)](https://codecov.io/gh/personalagile/wildlifescanner)

A lean, modular Python tool that watches an input directory for new videos, detects animal activity, and extracts only relevant segments into an output directory. Configuration is driven by a `.env` file in the input directory. Logs are written into the output directory.

## Features
- Directory watching (watchdog)
- Pluggable detectors (base interface); default: YOLO (Ultralytics)
- Segmentation with pre-roll, post-roll, minimum duration, and merge gap
- Video cutting via FFmpeg (stream copy with re-encode fallback)
- Clean logs, tests, and code quality (ruff, black, pytest)
- Optional post-processing: zoom (static crop via FFmpeg) and tracking (dynamic crop via OpenCV) with smoothing, deadzone, and optional aspect lock
- Configurable retention: `KEEP_POSTPROCESSED=true` keeps processed files with `_zoom`/`_track` suffix; otherwise originals are replaced

## Quick Start
```bash
make install
mkdir -p input output
# optional: seed defaults
cp .env.example input/.env
# drop MP4s into input/
make run INPUT=./input OUTPUT=./output
# or
python -m wildlifescanner --input input --output output
```

## Requirements
- Python 3.10+
- FFmpeg installed and available on PATH (`ffmpeg`)
- Optional: GPU for faster inference

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional for development
```

Alternatively, use the Makefile:
```bash
make install
```

## Usage
1. Create an input directory (e.g., `/data/wildlife/input`) and place a `.env` file there. See `.env.example`.
2. Set an output directory (can also be configured via `.env`).
3. Run the program:
```bash
python -m wildlifescanner --input /data/wildlife/input --output /data/wildlife/output
```

The program automatically loads `.env` from the input directory and applies those settings.

## Post-processing (zoom/tracking)
- Enable with `ZOOM_ENABLED=true` (static crop) or `TRACKING_ENABLED=true` (dynamic crop + smoothing).
- Minimum output resolution: `MIN_OUTPUT_WIDTH` x `MIN_OUTPUT_HEIGHT`.
- File retention: if `KEEP_POSTPROCESSED=true`, processed files are kept alongside originals with `_zoom`/`_track` suffix; otherwise the original segment is replaced.

### Examples

- __Zoom and replace originals__
  - `input/.env`:
  ```env
  ZOOM_ENABLED=true
  MIN_OUTPUT_WIDTH=960
  MIN_OUTPUT_HEIGHT=540
  KEEP_POSTPROCESSED=false
  ```
  - Run:
  ```bash
  python -m wildlifescanner --input input --output output
  ```
  - Result: segments like `VID_seg001_...MP4` are cropped/zoomed and the originals are replaced (no `_zoom` file kept).

- __Tracking and keep processed alongside originals__
  - `input/.env`:
  ```env
  TRACKING_ENABLED=true
  KEEP_POSTPROCESSED=true
  MIN_OUTPUT_WIDTH=960
  MIN_OUTPUT_HEIGHT=540
  # optional tuning
  TRACKING_CENTER_ALPHA=0.05
  TRACKING_SIZE_ALPHA=0.04
  TRACKING_MAX_MOVE_FRAC=0.05
  TRACKING_MAX_ZOOM_FRAC=0.06
  TRACKING_CENTER_DEADZONE_FRAC=0.10
  TRACKING_ZOOM_DEADZONE_FRAC=0.12
  TRACKING_MARGIN=0.20
  ```
  - Run:
  ```bash
  python -m wildlifescanner --input input --output output
  ```
  - Result: in addition to `VID_seg001_...MP4`, a `VID_seg001_..._track.MP4` is kept. If both `TRACKING_ENABLED` and `ZOOM_ENABLED` are `true`, tracking takes precedence.

## Configuration (.env in the input directory)
See `.env.example`. Important keys:
- INPUT_DIR: recommended via CLI, but can be set here
- OUTPUT_DIR: target directory for extracted segments and logs
- DETECTOR: "YOLO" (default) or future "MEGADETECTOR"
- YOLO_MODEL: path or model name (e.g., `yolov8n.pt`)
- CONFIDENCE_THRESHOLD: e.g., 0.25
- FRAME_STRIDE: e.g., 5 (infer every 5th frame)
- PREROLL_SEC, POSTROLL_SEC, MIN_ACTIVITY_SEC, MERGE_GAP_SEC
- ZOOM_ENABLED, TRACKING_ENABLED, KEEP_POSTPROCESSED
- MIN_OUTPUT_WIDTH, MIN_OUTPUT_HEIGHT
- Tracking smoothing (advanced): `TRACKING_CENTER_ALPHA`, `TRACKING_SIZE_ALPHA`,
  `TRACKING_MAX_MOVE_FRAC`, `TRACKING_MAX_ZOOM_FRAC`, `TRACKING_CENTER_DEADZONE_FRAC`,
  `TRACKING_ZOOM_DEADZONE_FRAC`, `TRACKING_MARGIN`

## Test video sets (positive/negative)
- Place your test videos here:
  - `tests/data/positive/` — videos containing animals
  - `tests/data/negative/` — videos without animals
- The integration test `tests/test_video_sets.py` will iterate over these directories and assert that animals are found in the positive set and not found in the negative set.
- These tests are disabled by default. Enable them by setting `RUN_VIDEO_TESTS=1`:
```bash
make video-tests
# or
RUN_VIDEO_TESTS=1 pytest -q -m "video_sets"
```

## Running tests
```bash
pytest
# with coverage
make cov
```

## Acceleration (Apple Silicon MPS / CUDA)
- Device selection is automatic: MPS (Apple) > CUDA > CPU. The chosen device is logged on startup by `YOLODetector`.
- Requirements for MPS: a PyTorch build with MPS support (default on Apple Silicon wheels).

Check MPS availability:
```python
import torch
print(torch.backends.mps.is_available())  # True means MPS is usable
```

Optional environment for mixed ops on macOS:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
This lets unsupported operators fall back to CPU instead of failing.

## Notes on species coverage
The default (COCO) model detects a subset of classes (e.g., bird, dog, cat, horse, sheep, cow, elephant, bear, zebra, giraffe). For species common in Germany (e.g., fox, wild boar, roe deer), consider MegaDetector or a fine-tuned YOLO model. The detector interface is modular to allow easy replacement.

## Architecture
See `architecture.md` for an arc42-style overview including ASCII diagrams of the system context, building blocks, and runtime flow.

## License
MIT
