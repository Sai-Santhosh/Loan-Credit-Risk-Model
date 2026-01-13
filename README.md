# 🏦 Credit Risk Prediction Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![AWS](https://img.shields.io/badge/AWS-Cloud%20Native-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black.svg)

**Production-grade machine learning pipeline for credit risk assessment**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [AWS Deployment](#-aws-deployment) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

This repository contains a comprehensive, production-ready ML pipeline for predicting credit risk in loan applications. Built with enterprise-grade standards, it features end-to-end automation from data ingestion to model deployment, leveraging AWS services and MLOps best practices.

### 🎯 Business Problem

Predict whether a loan applicant is likely to **default** (Charged Off) or **fully repay** (Fully Paid) their loan, enabling:
- Automated credit decisioning
- Risk-based pricing
- Portfolio risk monitoring
- Regulatory compliance (Fair Lending)

### 📊 Model Performance

| Metric | LightGBM | XGBoost |
|--------|----------|---------|
| AUC-ROC | **0.718** | 0.724 |
| Accuracy | 69.4% | 66.4% |
| Precision | 88% | 89% |
| Recall | 72% | 67% |
| F1 Score | 0.79 | 0.76 |

---

## ✨ Features

### 🔧 Core Capabilities

- **End-to-End ML Pipeline**: Data ingestion → Preprocessing → Training → Evaluation → Deployment
- **Multiple Model Support**: LightGBM and XGBoost with hyperparameter tuning
- **Model Interpretability**: SHAP-based feature importance and prediction explanations
- **Production Code Standards**: OOP design, type hints, comprehensive logging, unit tests

### ☁️ AWS Integration

| Service | Purpose |
|---------|---------|
| **S3** | Data lake for raw/processed data and model artifacts |
| **Lambda** | Real-time inference API and data ingestion triggers |
| **Glue** | ETL jobs for large-scale data processing |
| **Athena** | SQL queries on data lake |
| **SageMaker** | Model training and endpoint deployment |
| **Redshift** | Data warehousing and analytics |
| **CloudFormation** | Infrastructure as Code |
| **Step Functions** | ML workflow orchestration |

### 📈 MLOps Features

- **Experiment Tracking**: MLflow with DagsHub integration
- **Model Registry**: Version control for models with staging/production promotion
- **Feature Store**: Offline/online feature serving
- **Monitoring**: Prometheus metrics and CloudWatch integration

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│  (CSV Files, APIs, Databases, Streaming)                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS S3 (Data Lake)                           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐                │
│  │ Raw Data │  │ Processed    │  │ Artifacts  │                │
│  │          │  │ Data         │  │ & Models   │                │
│  └──────────┘  └──────────────┘  └────────────┘                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ AWS Glue  │  │ AWS       │  │ AWS       │
│ ETL Jobs  │  │ Athena    │  │ Redshift  │
└───────────┘  └───────────┘  └───────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML TRAINING PIPELINE                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ Feature    │  │ Model      │  │ HPO with   │                │
│  │ Engineering│─▶│ Training   │─▶│ Optuna     │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│         │                               │                       │
│         ▼                               ▼                       │
│  ┌────────────┐                 ┌────────────┐                 │
│  │ MLflow/    │                 │ Model      │                 │
│  │ DagsHub    │                 │ Registry   │                 │
│  └────────────┘                 └────────────┘                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT & SERVING                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ AWS Lambda │  │ SageMaker  │  │ API        │                │
│  │ Functions  │  │ Endpoints  │  │ Gateway    │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
credit-risk-prediction/
│
├── 📁 config/                    # Configuration files
│   ├── config.yaml              # Main configuration
│   └── logging_config.py        # Logging setup
│
├── 📁 src/                       # Source code
│   ├── data/                    # Data loading & transformation
│   │   ├── data_loader.py       # S3, Redshift, local loaders
│   │   ├── data_validator.py    # Schema validation
│   │   └── data_transformer.py  # Preprocessing pipeline
│   ├── features/                # Feature engineering
│   │   ├── feature_engineering.py
│   │   └── feature_store.py
│   ├── models/                  # Model implementations
│   │   ├── base_model.py
│   │   ├── lgbm_model.py
│   │   ├── xgboost_model.py
│   │   └── model_registry.py
│   ├── training/                # Training utilities
│   │   ├── trainer.py           # MLflow-integrated trainer
│   │   └── hyperparameter_tuner.py
│   ├── evaluation/              # Model evaluation
│   │   ├── evaluator.py
│   │   └── shap_analyzer.py
│   └── pipeline/                # End-to-end pipelines
│       ├── training_pipeline.py
│       └── inference_pipeline.py
│
├── 📁 aws/                       # AWS infrastructure
│   ├── lambda/                  # Lambda function handlers
│   ├── glue/                    # Glue ETL scripts
│   ├── sagemaker/               # SageMaker jobs
│   └── cloudformation/          # IaC templates
│
├── 📁 notebooks/                 # Jupyter notebooks
│   ├── 00_unclean_original.ipynb
│   ├── 01_data_ingestion.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_shap_analysis.ipynb
│
├── 📁 scripts/                   # CLI scripts
│   ├── train.py
│   ├── evaluate.py
│   └── run_pipeline.py
│
├── 📁 tests/                     # Test suite
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_pipeline.py
│
├── 📁 data/                      # Data directories
│   ├── raw/
│   ├── processed/
│   └── artifacts/
│
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-container setup
├── Makefile                     # Build automation
├── requirements.txt             # Python dependencies
└── setup.py                     # Package configuration
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- AWS Account (for cloud deployment)
- DagsHub Account (for experiment tracking)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/credit-risk-prediction.git
cd credit-risk-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env  # Add your AWS credentials and DagsHub token
```

### Running the Pipeline

```bash
# Using Makefile
make train                    # Train model
make evaluate                 # Evaluate model
make run-pipeline             # Full pipeline

# Using Python scripts
python scripts/train.py --model lightgbm --tune
python scripts/evaluate.py --shap
python scripts/run_pipeline.py --dagshub-repo username/credit-risk
```

### Using Docker

```bash
# Build and run
docker-compose up -d

# Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Jupyter: http://localhost:8888
```

---

## ☁️ AWS Deployment

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file aws/cloudformation/infrastructure.yaml \
    --stack-name credit-risk-ml-pipeline \
    --parameter-overrides Environment=production \
    --capabilities CAPABILITY_NAMED_IAM

# Upload Glue scripts
aws s3 cp aws/glue/etl_job.py s3://credit-risk-ml-pipeline/scripts/glue/

# Deploy Lambda functions
make aws-lambda
```

### Run ETL Pipeline

```bash
# Trigger Glue job
aws glue start-job-run --job-name credit-risk-etl-job

# Check status
aws glue get-job-run --job-name credit-risk-etl-job --run-id <run-id>
```

### Deploy Model to SageMaker

```bash
# Train on SageMaker
python aws/sagemaker/training_job.py --instance-type ml.m5.xlarge

# Deploy endpoint
python aws/sagemaker/deploy_endpoint.py
```

---

## 📊 MLflow & DagsHub Integration

### Setup DagsHub

```python
import dagshub
import mlflow

# Initialize DagsHub
dagshub.init(repo_owner="your-username", repo_name="credit-risk", mlflow=True)

# MLflow will now track to DagsHub
mlflow.set_experiment("credit-risk-prediction")
```

### Track Experiments

```bash
# Training with MLflow tracking
python scripts/train.py \
    --dagshub-repo your-username/credit-risk \
    --experiment-name credit-risk-v2

# View experiments
mlflow ui
# Or visit: https://dagshub.com/your-username/credit-risk.mlflow
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific tests
pytest tests/test_models.py -v
```

---

## 📈 Model Features

### Input Features (19 total)

| Feature | Type | Description |
|---------|------|-------------|
| `term` | Categorical | Loan term (36 or 60 months) |
| `int_rate` | Numerical | Interest rate |
| `grade` | Categorical | Loan grade (A-G) |
| `emp_length` | Categorical | Employment length |
| `home_ownership` | Categorical | Home ownership status |
| `annual_inc` | Numerical | Annual income |
| `verification_status` | Categorical | Income verification status |
| `purpose` | Categorical | Loan purpose |
| `dti` | Numerical | Debt-to-income ratio |
| `revol_util` | Numerical | Revolving utilization |
| `loan_to_income` | Engineered | Loan amount / Annual income |
| `total_interest_owed` | Engineered | Loan × Interest rate |
| `installment_to_income_ratio` | Engineered | Payment / Monthly income |
| `active_credit_pct` | Engineered | Open accounts / Total accounts |
| `credit_age` | Engineered | Years since first credit |

### Top SHAP Feature Importance

1. **DTI** - Debt-to-income ratio
2. **Revolving Utilization** - Credit utilization
3. **Interest Rate** - Loan interest rate
4. **Annual Income** - Borrower income
5. **Active Credit %** - Credit account ratio

---

## 🔒 Security

- All credentials stored in environment variables
- AWS IAM roles with least-privilege access
- S3 bucket encryption enabled
- VPC endpoints for private connectivity
- API authentication via API keys

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Data Science Team**
- Email: datascience@company.com
- GitHub: [@your-username](https://github.com/your-username)

---

<div align="center">

**Built with ❤️ for the Financial Services Industry**

*Transforming complex banking processes into intelligent software solutions*

</div>
