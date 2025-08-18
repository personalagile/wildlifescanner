# Simple project Makefile

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
BLACK := .venv/bin/black
MYPY := .venv/bin/mypy

INPUT ?= ./input
OUTPUT ?= ./output
DETECTOR ?= YOLO

.PHONY: help
help:
	@echo "Targets: venv, install, format, lint, typecheck, test, cov, run, clean"

.PHONY: venv
venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

.PHONY: install
install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

.PHONY: format
format:
	$(RUFF) format .
	$(BLACK) .

.PHONY: lint
lint:
	$(RUFF) check .

.PHONY: typecheck
typecheck:
	$(MYPY) wildlifescanner

.PHONY: test
test:
	PYTHONPATH=. $(PYTEST) -q

.PHONY: cov
cov:
	PYTHONPATH=. $(PYTEST) --cov=wildlifescanner --cov-report=term-missing

.PHONY: video-tests
video-tests:
	PYTHONPATH=. RUN_VIDEO_TESTS=1 $(PYTEST) -q -m "video_sets"

.PHONY: run
run:
	$(PYTHON) -m wildlifescanner --input $(INPUT) --output $(OUTPUT) --detector $(DETECTOR)

.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
