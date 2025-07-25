# Makefile for XAI Medical Images project

.PHONY: help install install-dev test test-cov lint format type-check clean run docker-build docker-run setup-dev

# Default target
help:
	@echo "Available commands:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  test           Run tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo "  clean          Clean up temporary files"
	@echo "  run            Run the Flask application"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo "  setup-dev      Setup development environment"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-watch:
	pytest-watch tests/ src/

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100
	bandit -r src/ -f json -o security-report.json || true

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

type-check:
	mypy src/ --ignore-missing-imports

format-check:
	black src/ tests/ --check --line-length=100
	isort src/ tests/ --profile black --check-only

# Security
security-check:
	safety check
	bandit -r src/

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .tox/
	rm -f security-report.json

# Application
run:
	python run.py

run-dev:
	ENVIRONMENT=development python run.py

run-prod:
	ENVIRONMENT=production gunicorn --bind 0.0.0.0:5000 --workers 4 run:app

# Training
train:
	python src/train.py --data_dir data/processed --epochs 50 --batch_size 32

train-sample:
	python src/train.py --data_dir data/sample --epochs 5 --batch_size 8

# Docker
docker-build:
	docker build -t xai-medical-images .

docker-run:
	docker run -p 5000:5000 xai-medical-images

docker-run-dev:
	docker run -p 5000:5000 -v $(PWD):/app -e ENVIRONMENT=development xai-medical-images

# Data management
validate-data:
	python scripts/validate_data.py --data_dir data/raw

process-data:
	python scripts/process_data.py --input_dir data/raw --output_dir data/processed

download-sample:
	python scripts/download_sample_data.py --output_dir data/sample

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Setup
setup-dev: install-dev
	mkdir -p logs data/raw data/processed data/sample models/checkpoints static/uploads
	touch logs/.gitkeep data/raw/.gitkeep data/processed/.gitkeep
	@echo "Development environment setup complete!"

setup-dirs:
	mkdir -p logs data/raw data/processed data/sample models/checkpoints static/uploads

# CI/CD
ci: install-dev format-check lint type-check test-cov

# Performance
profile:
	python -m cProfile -o profile.stats run.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

benchmark:
	python scripts/benchmark_model.py

# Database (if needed in future)
db-init:
	python scripts/init_db.py

db-migrate:
	python scripts/migrate_db.py

# Monitoring
logs:
	tail -f logs/app.log

health-check:
	curl -f http://localhost:5000/health || exit 1

# Deployment helpers
deploy-staging:
	@echo "Deploying to staging..."
	# Add staging deployment commands here

deploy-prod:
	@echo "Deploying to production..."
	# Add production deployment commands here

# Version management
version:
	@python -c "import pkg_resources; print(pkg_resources.get_distribution('XAI-Medical-Images').version)" 2>/dev/null || echo "Not installed as package"

# Quick development workflow
dev: clean install-dev format lint test
	@echo "Development workflow complete!"

# All quality checks
check-all: format-check lint type-check security-check test-cov
	@echo "All quality checks passed!"