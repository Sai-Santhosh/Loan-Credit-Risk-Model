# =============================================================================
# Credit Risk Prediction - Makefile
# Production-grade build automation
# =============================================================================

.PHONY: help install install-dev clean lint format test test-cov run-api train evaluate deploy docker-build docker-run aws-deploy

# Default target
help:
	@echo "Credit Risk Prediction - Available Commands"
	@echo "============================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make setup-pre-commit Setup pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linting checks"
	@echo "  make format           Format code with black & isort"
	@echo "  make type-check       Run mypy type checking"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make train            Train the model"
	@echo "  make evaluate         Evaluate model performance"
	@echo "  make tune             Run hyperparameter tuning"
	@echo "  make shap             Generate SHAP analysis"
	@echo ""
	@echo "API & Deployment:"
	@echo "  make run-api          Run the FastAPI server locally"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo ""
	@echo "AWS Deployment:"
	@echo "  make aws-deploy       Deploy to AWS"
	@echo "  make aws-glue-job     Deploy Glue ETL job"
	@echo "  make aws-lambda       Deploy Lambda functions"
	@echo "  make aws-sagemaker    Deploy SageMaker endpoint"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts"
	@echo "  make clean-all        Remove all generated files"

# =============================================================================
# Environment Setup
# =============================================================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

setup-pre-commit:
	pre-commit install
	pre-commit autoupdate

create-dirs:
	mkdir -p data/raw data/processed data/artifacts
	mkdir -p models logs
	mkdir -p mlruns

# =============================================================================
# Code Quality
# =============================================================================

lint:
	flake8 src/ tests/ scripts/ --max-line-length=100 --ignore=E203,W503
	pylint src/ --max-line-length=100 --disable=C0114,C0115,C0116

format:
	black src/ tests/ scripts/ --line-length=100
	isort src/ tests/ scripts/ --profile=black --line-length=100

type-check:
	mypy src/ --ignore-missing-imports --no-strict-optional

check-all: lint type-check
	@echo "All checks passed!"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-aws:
	pytest tests/aws/ -v --tb=short -m aws

# =============================================================================
# ML Pipeline
# =============================================================================

train:
	python scripts/train.py --config config/config.yaml

evaluate:
	python scripts/evaluate.py --config config/config.yaml

tune:
	python scripts/hyperparameter_tuning.py --config config/config.yaml

shap:
	python scripts/shap_analysis.py --config config/config.yaml

run-pipeline:
	python scripts/run_pipeline.py --config config/config.yaml

# =============================================================================
# Notebooks
# =============================================================================

run-notebooks:
	papermill notebooks/01_data_ingestion.ipynb notebooks/output/01_data_ingestion_output.ipynb
	papermill notebooks/02_exploratory_data_analysis.ipynb notebooks/output/02_eda_output.ipynb
	papermill notebooks/03_data_preprocessing.ipynb notebooks/output/03_preprocessing_output.ipynb

jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# =============================================================================
# API & Local Deployment
# =============================================================================

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-api-prod:
	gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t credit-risk-prediction:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env credit-risk-prediction:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# =============================================================================
# AWS Deployment
# =============================================================================

aws-deploy:
	@echo "Deploying to AWS..."
	python scripts/deploy.py --target all

aws-s3-sync:
	aws s3 sync data/processed/ s3://$(S3_BUCKET_NAME)/data/processed/
	aws s3 sync models/ s3://$(S3_BUCKET_NAME)/models/

aws-glue-job:
	@echo "Deploying Glue ETL job..."
	aws s3 cp aws/glue/etl_job.py s3://$(S3_BUCKET_NAME)/scripts/glue/
	aws glue create-job --name credit-risk-etl --role $(GLUE_JOB_ROLE) \
		--command "Name=glueetl,ScriptLocation=s3://$(S3_BUCKET_NAME)/scripts/glue/etl_job.py"

aws-lambda:
	@echo "Deploying Lambda functions..."
	cd aws/lambda && zip -r lambda_package.zip . && \
	aws lambda update-function-code --function-name credit-risk-inference \
		--zip-file fileb://lambda_package.zip

aws-sagemaker:
	@echo "Deploying SageMaker endpoint..."
	python aws/sagemaker/deploy_endpoint.py

aws-step-functions:
	@echo "Deploying Step Functions workflow..."
	aws stepfunctions create-state-machine \
		--name credit-risk-ml-pipeline \
		--definition file://aws/step_functions/ml_pipeline_definition.json \
		--role-arn $(STEP_FUNCTIONS_ROLE)

# =============================================================================
# MLflow / DagsHub
# =============================================================================

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

dagshub-setup:
	dagshub login
	python -c "import dagshub; dagshub.init(repo_owner='$(DAGSHUB_REPO_OWNER)', repo_name='$(DAGSHUB_REPO_NAME)', mlflow=True)"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .eggs/

clean-all: clean
	rm -rf data/processed/* data/artifacts/*
	rm -rf models/* logs/*
	rm -rf mlruns/ mlartifacts/
	rm -rf htmlcov/ .coverage
