.PHONY: setup install test lint format clean download-data

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install:
	pip install -r requirements.txt

download-data:
	python3 scripts/download_criteo.py

test:
	pytest -v --cov=src tests/

lint:
	flake8 src tests
	isort --check-only src tests
	black --check src tests

format:
	isort src tests
	black src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov
