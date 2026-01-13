"""
Credit Risk Prediction - Production ML Pipeline
================================================

A production-grade machine learning pipeline for credit risk assessment,
featuring AWS integration, MLflow experiment tracking, and enterprise-ready
model deployment capabilities.

Author: Data Science Team
Version: 1.0.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="credit-risk-prediction",
    version="1.0.0",
    author="Data Science Team",
    author_email="datascience@company.com",
    description="Production-grade ML pipeline for credit risk prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/credit-risk-prediction",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/credit-risk-prediction/issues",
        "Documentation": "https://github.com/your-org/credit-risk-prediction#readme",
        "Source Code": "https://github.com/your-org/credit-risk-prediction",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "credit-risk-train=scripts.train:main",
            "credit-risk-evaluate=scripts.evaluate:main",
            "credit-risk-deploy=scripts.deploy:main",
            "credit-risk-pipeline=scripts.run_pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
