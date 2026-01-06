# 🍷 Wine Quality MLOps Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688.svg)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-945dd6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg)](https://mlflow.org/)
[![W&B](https://img.shields.io/badge/W%26B-Visualization-FFBE00.svg)](https://wandb.ai/)
[![Monitoring](https://img.shields.io/badge/Grafana_Cloud-OpenTelemetry-F46800.svg)](https://grafana.com/)

An end-to-end production-grade Machine Learning pipeline for wine quality prediction, featuring robust data versioning, experiment tracking, and real-time observability.

---

## 🚀 Key Features

*   **Production API**: Built with **FastAPI** using an optimized prediction pipeline (singleton model loading).
*   **Observability**: Real-time metric streaming to **Grafana Cloud** via **OpenTelemetry (OTLP)**.
*   **Drift Detection**: Automated data drift analysis using **Evidently AI** with interactive HTML reporting.
*   **Experiment Management**: Multi-platform tracking using **MLflow** (remote via Dagshub) and **Weights & Biases**.
*   **Automatic EDA**: Instant data profiling and visualization using **Sweetviz**.
*   **Load Testing**: Integrated **Locust** suite with a custom "Drift Attack" toggle to verify monitoring reliability.
*   **Reproducibility**: Entire environment managed via **UV** for lighting-fast, deterministic builds.

---

## 🛠 Tech Stack

| Domain | Tools |
| :--- | :--- |
| **Framework** | FastAPI, Uvicorn, Jinja2 |
| **Data/ML** | Scikit-Learn, Pandas, DVC, Joblib |
| **Tracking** | MLflow, Dagshub, W&B |
| **Monitoring** | Grafana Cloud, OpenTelemetry, Prometheus |
| **Analysis** | Evidently AI, Sweetviz |
| **Infrastructure** | Docker, Docker-compose, UV |

---

## 🏗️ MLOps Toolbox: Roles & Responsibilities

This project integrates a diverse set of industry-standard tools, each serving a specific role in the MLOps lifecycle:

*   **DVC (Data Version Control)**: The "Git for Data." It versions large datasets and model files, allowing us to roll back data just like we roll back code.
*   **MLflow**: Our experiment headquarters. It records every training run, parameter, and metric, ensuring we never lose track of a successful model.
*   **Dagshub**: The central hub that hosts our DVC data and MLflow experiments, making them accessible from anywhere.
*   **Weights & Biases (W&B)**: Provides advanced visualizations of model training and feature importance for deep performance analysis.
*   **FastAPI**: The fast, modern "engine" that hosts our model and serves predictions via a professional REST API.
*   **Evidently AI**: The "Security Guard" for our data. It monitors live traffic to detect **Data Drift**, alerting us when the real-world data starts to diverge from our training set.
*   **Sweetviz**: Our automated EDA detective. It generates instant, interactive reports to help us understand the statistical properties of our data.
*   **OpenTelemetry & Grafana**: Our high-tech dashboarding stack. It streams real-time health metrics from our local API directly to the cloud.
*   **Locust**: The "Chaos Monkey" used to simulate heavy user traffic and intentional data drift attacks to test our system's resilience.
*   **UV**: The lightning-fast package manager that ensures our environment is set up in seconds with 100% reproducibility.

---

## 📁 Project Structure

```text
├── artifacts/             # Data, models, and evaluation reports
├── config/                # Pipeline configuration (YAML)
├── src/mlProject/         # Core package logic
│   ├── components/        # Pipeline stage implementations
│   ├── pipeline/          # Orchestration scripts
│   └── entity/            # Pydantic/Config data classes
├── templates/             # Web UI HTML files
├── app.py                 # Main FastAPI Application
├── locustfile.py          # Load test suite
├── dvc.yaml               # DVC pipeline definitions
└── main.py                # Pipeline execution entry point
```

---

## ⚡ Quick Start

### 1. Requirements
*   Python 3.12+
*   [UV](https://github.com/astral-sh/uv) (Recommended for speed)

### 2. Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd EtoE

# Create environment and install dependencies
uv sync
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
# Grafana / OTLP Setup
OTLP_ENDPOINT="your-grafana-otlp-endpoint"
GRAFANA_USER="your-stack-id"
GRAFANA_TOKEN="your-token"

# MLflow / Dagshub (Optional)
MLFLOW_TRACKING_URI="https://dagshub.com/..."
MLFLOW_TRACKING_USERNAME="..."
MLFLOW_TRACKING_PASSWORD="..."
```

---

---

## 🏃 Quick Execution

To get the system up and running immediately, follow these two steps:

### 1️⃣ Start the API Server
Open your terminal and run:
```bash
source .venv/bin/activate
python app.py
```
> **🔗 Access**: [http://localhost:8080](http://localhost:8080)

### 2️⃣ Start Locust (Load Testing & Drift Attack)
Open a **new** terminal tab and run:
```bash
source .venv/bin/activate
locust -f locustfile.py --host http://localhost:8080
```
> **🔗 Access**: [http://localhost:8089](http://localhost:8089)

---

## 🖥️ Detailed Dashboard Guide

Once the services are running, you can explore these specialized views:

### 🔍 Analysis Dashboards
*   **Automated EDA (Sweetviz)**: [http://localhost:8080/data_profiling](http://localhost:8080/data_profiling)
    *   *Deep dive into training data distributions and correlations.*
*   **Live Drift Report (Evidently)**: [http://localhost:8080/drift_report](http://localhost:8080/drift_report)
    *   *Real-time comparison of production traffic vs. baseline training data.*

### 🛠️ Drift Simulation Steps
1.  Go to the **Locust UI** ([localhost:8089](http://localhost:8089)).
2.  Start a swarm with **10 users**.
3.  Under **Advanced Options**, check ✅ **"Enable simulated data drift"**.
4.  Refresh the **Drift Report** to see "Alcohol" drift in real-time!

### ☁️ External MLOps Hubs
Aside from the local dashboards, the pipeline is integrated with these external platforms for advanced tracking:

*   **Experiment Tracking (MLflow)**: Log into your [Dagshub Repository](https://dagshub.com/) to view the MLflow UI.
    *   *Role*: Track hyperparameters, loss curves, and model versions.
*   **Model Visualization (W&B)**: Visit your [W&B Project Dashboard](https://wandb.ai/).
    *   *Role*: Deep performance analysis and feature importance visualization.
*   **Production Telemetry (Grafana)**: Check your [Grafana Cloud Explore view](https://grafana.com/).
    *   *Role*: Monitor live system health metrics like `data_drift_score`.

> **💡 Security Note**: This project uses a `.env` file (which is git-ignored) to store your API keys. If someone clones this project, they will need to provide their **own** keys to see data on these dashboards. Your data and accounts remain private and will not be cluttered by other users' runs.

---

## 📈 Monitoring Workflow

1.  **Prediction**: Every API request is logged to `artifacts/predictions/inference_log.csv`.
2.  **Telemetry**: Real-time metrics (request counts, latency) are sent to Grafana.
3.  **Drift Check**: The `/check_drift` endpoint compares current inference data against the training baseline.
4.  **Visualize**: Use `/drift_report` to identify which features (e.g., Alcohol) are causing model performance degradation.

---

## 🤝 Contributing
Feel free to open issues or submit pull requests for features like Retraining Triggers, CI/CD pipelines, or Hyperparameter tuning improvements.