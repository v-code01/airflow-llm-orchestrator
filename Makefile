.PHONY: help install install-dev clean test test-cov lint format check security docs build publish

PYTHON := python3
PIP := pip
PROJECT_NAME := airflow-llm-orchestrator

help:
	@echo "Available commands:"
	@echo "  install      Install package"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code"
	@echo "  check        Run all checks (lint, format, test, security)"
	@echo "  security     Run security checks"
	@echo "  docs         Build documentation"
	@echo "  build        Build package"
	@echo "  publish      Publish to PyPI"
	@echo "  setup-dev    Set up development environment"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]
	pre-commit install

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=airflow_llm --cov-report=html --cov-report=term-missing

lint:
	flake8 airflow_llm tests
	pylint airflow_llm
	mypy airflow_llm

format:
	black airflow_llm tests
	isort airflow_llm tests

check: lint test security
	@echo "All checks passed!"

security:
	bandit -r airflow_llm
	safety check
	pip-audit

docs:
	cd docs && make html

build: clean
	$(PYTHON) -m build

publish: build
	$(PYTHON) -m twine upload dist/*

setup-dev: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make check' to verify everything is working."

pre-commit:
	pre-commit run --all-files

tox:
	tox

benchmark:
	$(PYTHON) -m pytest tests/ -k "benchmark" --benchmark-only

profile:
	$(PYTHON) -m cProfile -o profile.stats -m pytest tests/
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

docker-build:
	docker build -t $(PROJECT_NAME):latest .

docker-test:
	docker run --rm -v $(PWD):/app -w /app $(PROJECT_NAME):latest make test

validate-release:
	$(PYTHON) setup.py check --strict --metadata --restructuredtext
	check-manifest
	pyroma .
