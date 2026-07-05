from fastapi import FastAPI
from app.schemas import WineInput, PredictionResponse
from app.predictor import predict_wine_quality, model_name


app = FastAPI(
    title="API de Clasificacion de Vinos",
    description="API para clasificar vinos como estándar o de alta calidad usando Machine Learning.",
    version="1.0.0"
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


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: WineInput):
    try:
        return predict_wine_quality(input_data.model_dump())
    except Exception as e:
        return {
            "model": "error",
            "prediction": -1,
            "quality_label": str(e),
            "probability_high_quality": 0,
            "threshold_used": 0
        }