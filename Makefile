.PHONY: setup install test lint format clean download-data download-attribution-data popularity-baseline cf-experiment ctr-experiment pre-serve-ctr-experiment ranking-integration-experiment

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

download-attribution-data:
	python3 scripts/download_attribution.py

popularity-baseline:
	python3 scripts/run_popularity_baseline.py

cf-experiment:
	python3 scripts/run_collaborative_filtering_experiment.py

ctr-experiment:
	python3 scripts/run_ctr_experiment.py

pre-serve-ctr-experiment:
	python3 scripts/run_pre_serve_ctr_experiment.py

ranking-integration-experiment:
	python3 scripts/run_ranking_integration_experiment.py

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
