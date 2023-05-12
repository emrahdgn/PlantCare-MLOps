# PlantCare-MLOps

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=data-version-control&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-F42535?style=for-the-badge&logo=mlflow&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-007ACC?style=for-the-badge&logo=optuna&logoColor=white)
![Great Expectations](https://img.shields.io/badge/Great%20Expectations-FFC107?style=for-the-badge&logo=great-expectations&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)



## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Upcoming Features](#upcoming-features)
- [Deep Learning Model](#deep-learning-model)
- [Deployment Instructions](#deployment-instructions)
- [Performance Metrics](#performance-metrics)

---

## Project Overview
PlantCare-MLOps is a comprehensive machine learning project that provides a health check service for plant leaves. By leveraging computer vision techniques, this project aims to automate the diagnosis of diseases in plant leaves, saving time and resources.

The project utilizes the dataset from the [Plant Pathology 2021 - FGVC8 competition](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/overview) available on Kaggle.

## Technologies Used
The following technologies are used in this project:

- PyTorch: Deep learning framework for training the computer vision model
- DVC (Data Version Control): Version control system for managing data and models on Amazon S3 storage
- MLflow: Experiment tracking and management tool
- Optuna: Framework for hyperparameter optimization
- Great Expectations: Library for testing and validating data
- FastAPI: Web framework for creating API endpoints
- Docker: Containerization tool for packaging the application
- Git CI: Continuous integration for automated testing and deployment

In addition to the above technologies, the following tools are used for code formatting, linting, and documentation:

- black: Code formatter following PEP8 style guide
- isort: Import statement formatter
- flake8: Linter for enforcing PEP8 style guide
- Typer: Library for building CLI applications
- pre-commit: Git hooks for running checks before commits
- mkdocs: Static site generator for creating documentation
- pytest-cov: Plugin for measuring test coverage
<p align="right">(<a href="#plantcare-mlops">back to top</a>)</p>

## Upcoming Features
The following features are planned to be added in future commits:

- Custom Great Expectations tests for data validation
- Model tests for evaluating model performance
- Git CD: Continuous deployment for automating the deployment process
- Streamlit: Interactive dashboard and visualization tool
- NVIDIA Triton: Framework for serving machine learning models in production
- TensorRT: Library for optimizing models for deployment on NVIDIA GPUs
- Jit-Compiling: Compilation of code for improved runtime performance
<p align="right">(<a href="#plantcare-mlops">back to top</a>)</p>

## Deep Learning Model
For this multi-label classification project, the EfficientNet model is used. 

EfficientNets are a family of convolutional neural network models that have achieved state-of-the-art performance on various image classification tasks. They are designed to provide high accuracy while being computationally efficient.
<p align="right">(<a href="#plantcare-mlops">back to top</a>)</p>

## Deployment Instructions
To deploy the project locally, follow the instructions below:

```bash
git clone https://github.com/emrahdgn/PlantCare-MLOps.git
cd PlantCare-MLOps
make venv
source venv/bin/activate
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app
```
<p align="right">(<a href="#plantcare-mlops">back to top</a>)</p>

## Performance Metrics
In this section only Micro and Macro performance metrics are presentend for the sake of simplicity.

| Metric              | Micro        | Macro        |
|---------------------|--------------|--------------|
| Precision           | 0.910        | 0.903        |
| Recall              | 0.884        | 0.871        |
| F1-Score            | 0.897        | 0.886        |
<p align="right">(<a href="#plantcare-mlops">back to top</a>)</p>


