# WildlifeScanner

 [![CI](https://github.com/personalagile/wildlifescanner/actions/workflows/ci.yml/badge.svg)](https://github.com/personalagile/wildlifescanner/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/personalagile/wildlifescanner/branch/main/graph/badge.svg)](https://codecov.io/gh/personalagile/wildlifescanner)

A lean, modular Python tool that watches an input directory for new videos, detects animal activity, and extracts only relevant segments into an output directory. Configuration is driven by a `.env` file in the input directory. Logs are written into the output directory.

## Features
- Directory watching (watchdog)
- Pluggable detectors (base interface); default: YOLO (Ultralytics)
- Segmentation with pre-roll, post-roll, minimum duration, and merge gap
- Video cutting via FFmpeg (stream copy with re-encode fallback)
- Clean logs, tests, and code quality (ruff, black, pytest)
- Prepared interface for future dynamic cropping/tracking

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

## Configuration (.env in the input directory)
See `.env.example`. Important keys:
- INPUT_DIR: recommended via CLI, but can be set here
- OUTPUT_DIR: target directory for extracted segments and logs
- DETECTOR: "YOLO" (default) or future "MEGADETECTOR"
- YOLO_MODEL: path or model name (e.g., `yolov8n.pt`)
- CONFIDENCE_THRESHOLD: e.g., 0.25
- FRAME_STRIDE: e.g., 5 (infer every 5th frame)
- PREROLL_SEC, POSTROLL_SEC, MIN_ACTIVITY_SEC, MERGE_GAP_SEC

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
