from app.schemas import WineInput, PredictionResponse
from app.predictor import predict_wine_quality, model_name
from fastapi import FastAPI, HTTPException
from app.metrics import get_metrics
from contextlib import asynccontextmanager
from app.alerts import get_alert_status

from app.monitoring import (
    start_request,
    finish_request,
    fail_request,
    process_request
)

from app.mlflow_tracking import (
    initialize_mlflow,
    close_mlflow_run
)

from app.drift import (
    initialize_drift_detector,
    get_drift_status
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    initialize_drift_detector()

    initialize_mlflow(
        model_name=model_name
    )

    try:
        yield
    finally:
        close_mlflow_run()


app = FastAPI(
    title="API de Clasificacion de Vinos",
    description="API para clasificar vinos como estándar o de alta calidad usando Machine Learning.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "API de Clasificacion de Vinos",
        "model": model_name,
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": model_name
    }


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.get("/alerts")
def alerts():
    return get_alert_status()


@app.get("/drift")
def drift_status():
    return get_drift_status()


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: WineInput):
    
    input_dictionary = input_data.model_dump()

    request_monitoring = start_request(
        input_data=input_dictionary
    )

    try:

        prediction_result = predict_wine_quality(
            input_dictionary
        )

        finish_request(
            request_monitoring=request_monitoring,
            prediction=prediction_result["prediction"]
        )

        process_request(request_monitoring)

        return prediction_result

    except Exception as e:

        fail_request(
            request_monitoring=request_monitoring,
            error=str(e)
        )

        process_request(request_monitoring)

        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error al realizar la predicción."
        )