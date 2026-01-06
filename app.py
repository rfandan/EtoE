from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenTelemetry / Grafana Cloud Imports
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_fastapi_instrumentator import Instrumentator

# Evidently AI Imports
from evidently import Report
from evidently.presets import DataDriftPreset

# --- Standard App Setup ---
app = FastAPI(title="Wine Quality Prediction API")

# Initialize Prometheus Instrumentator to expose /metrics locally
Instrumentator().instrument(app).expose(app)

# --- Grafana Cloud / OpenTelemetry Setup ---
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT")
GRAFANA_USER = os.getenv("GRAFANA_USER")
GRAFANA_TOKEN = os.getenv("GRAFANA_TOKEN")

if OTLP_ENDPOINT and GRAFANA_USER and GRAFANA_TOKEN:
    import base64
    auth_header = base64.b64encode(f"{GRAFANA_USER}:{GRAFANA_TOKEN}".encode()).decode()
    
    resource = Resource(attributes={SERVICE_NAME: "wine-quality-api"})
    # For Grafana Cloud OTLP, the endpoint should point to /v1/metrics
    exporter = OTLPMetricExporter(
        endpoint=f"{OTLP_ENDPOINT}/v1/metrics",
        headers={"Authorization": f"Basic {auth_header}"}
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    meter = metrics.get_meter("wine.quality.meter")
    drift_gauge = meter.create_gauge(
        name="data_drift_score",
        description="Share of drifted features (0 to 1)",
    )
    print(f"OpenTelemetry initialized for Grafana Cloud (User: {GRAFANA_USER})")
else:
    print("Monitoring disabled: Missing OTLP_ENDPOINT, GRAFANA_USER, or GRAFANA_TOKEN")
    drift_gauge = None

templates = Jinja2Templates(directory="templates")

class WineFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fixed_acidity: float = Field(..., alias="fixed acidity", json_schema_extra={"example": 7.4})
    volatile_acidity: float = Field(..., alias="volatile acidity", json_schema_extra={"example": 0.7})
    citric_acid: float = Field(..., alias="citric acid", json_schema_extra={"example": 0.0})
    residual_sugar: float = Field(..., alias="residual sugar", json_schema_extra={"example": 1.9})
    chlorides: float = Field(..., json_schema_extra={"example": 0.076})
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide", json_schema_extra={"example": 11.0})
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide", json_schema_extra={"example": 34.0})
    density: float = Field(..., json_schema_extra={"example": 0.9978})
    pH: float = Field(..., json_schema_extra={"example": 3.51})
    sulphates: float = Field(..., json_schema_extra={"example": 0.56})
    alcohol: float = Field(..., json_schema_extra={"example": 9.4})

def calculate_drift():
    """Background task to calculate data drift using Evidently AI"""
    try:
        # 1. Load Reference Data (Training set)
        reference_data = pd.read_csv("artifacts/data_ingestion/data.csv")
        # Ensure columns match what we expect in prediction
        reference_data = reference_data.drop('quality', axis=1)

        # 2. Load Current Data (Inference Logs)
        log_path = "artifacts/predictions/inference_log.csv"
        if not os.path.exists(log_path):
            print("Drift check skipped: No inference logs found yet.")
            return
        
        current_data = pd.read_csv(log_path)
        # Drop non-feature columns
        current_data = current_data.drop(['prediction', 'timestamp'], axis=1)

        # 3. Run Evidently Drift Report
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference_data, current_data=current_data)
        
        # 4. Extract drift score (share of drifted features)
        result = snapshot.dict()
        drift_share = result['metrics'][0]['value']['share']
        
        # 5. Update OpenTelemetry Gauge
        if drift_gauge:
            drift_gauge.set(drift_share)
        print(f"Drift check completed. Drift Share: {drift_share}")

    except Exception as e:
        print(f"Error calculating drift: {e}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/check_drift")
async def check_drift(background_tasks: BackgroundTasks):
    background_tasks.add_task(calculate_drift)
    return {"message": "Drift calculation started in background"}

@app.get("/drift_report")
async def drift_report():
    """Generates and serves a full interactive Evidently AI drift report"""
    try:
        reference_data = pd.read_csv("artifacts/data_ingestion/data.csv")
        reference_data = reference_data.drop('quality', axis=1)

        log_path = "artifacts/predictions/inference_log.csv"
        if not os.path.exists(log_path):
            return HTMLResponse(content="<h1>No inference logs found yet. Run some predictions first!</h1>", status_code=404)
        
        current_data = pd.read_csv(log_path)
        current_data = current_data.drop(['prediction', 'timestamp'], axis=1)

        # Generate the interactive report
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference_data, current_data=current_data)
        
        report_path = "artifacts/predictions/drift_report.html"
        snapshot.save_html(report_path)
        
        return FileResponse(report_path)

    except Exception as e:
        return HTMLResponse(content=f"<h1>Error generating report: {str(e)}</h1>", status_code=500)

@app.get("/data_profiling")
async def data_profiling():
    """Serves the Sweetviz automated EDA report"""
    report_path = "artifacts/data_validation/report.html"
    if os.path.exists(report_path):
        return FileResponse(report_path)
    return HTMLResponse(content="<h1>Profiling report not found. Run the data validation pipeline first!</h1>", status_code=404)

# --- Prediction Pipeline Initialization ---
# Initialize the pipeline once at startup to avoid repeated disk reads
prediction_pipeline = PredictionPipeline()

@app.post("/predict")
async def predict_api(features: WineFeatures):
    try:
        data = [
            features.fixed_acidity, features.volatile_acidity, features.citric_acid,
            features.residual_sugar, features.chlorides, features.free_sulfur_dioxide,
            features.total_sulfur_dioxide, features.density, features.pH,
            features.sulphates, features.alcohol
        ]
        data = np.array(data).reshape(1, 11)
        
        # Use the global optimized pipeline
        predict = prediction_pipeline.predict(data)

        return {"prediction": float(predict[0])}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
    alcohol: float = Form(...)
):
    try:
        data = [
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]
        data = np.array(data).reshape(1, 11)
        
        # Use the global optimized pipeline
        predict = prediction_pipeline.predict(data)

        return templates.TemplateResponse("results.html", {"request": request, "prediction": float(predict[0])})
    except Exception as e:
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
