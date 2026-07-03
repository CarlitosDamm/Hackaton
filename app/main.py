import joblib
import pandas as pd
from fastapi import FastAPI
from schemas import WineInput


app = FastAPI(
    title="API de Clasificador de Vino",
    description="API para clasificar vinos como estándar o de alta calidad usando un modelo de Machine Learning.",
    version="1.0.0"
)

# Cargar modelo al iniciar la API
package = joblib.load("model/wine_classifier.joblib")

model = package["model"]
model_name = package["model_name"]
threshold = package["threshold"]
feature_names = package["feature_names"]
target_labels = package["target_labels"]


@app.get("/")
def root():
    return {
        "message": "API de Clasificador de Vino",
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


@app.post("/predict")
def predict(input_data: WineInput):
    # Convertir entrada JSON a diccionario
    data = input_data.model_dump()

    # Convertir nombres del API a nombres originales del dataset
    input_df = pd.DataFrame([{
        "fixed acidity": data["fixed_acidity"],
        "volatile acidity": data["volatile_acidity"],
        "citric acid": data["citric_acid"],
        "residual sugar": data["residual_sugar"],
        "chlorides": data["chlorides"],
        "free sulfur dioxide": data["free_sulfur_dioxide"],
        "total sulfur dioxide": data["total_sulfur_dioxide"],
        "density": data["density"],
        "pH": data["pH"],
        "sulphates": data["sulphates"],
        "alcohol": data["alcohol"]
    }])

    # Asegurar el orden correcto de columnas
    input_df = input_df[feature_names]

    # Obtener probabilidad de clase 1
    probability = model.predict_proba(input_df)[0][1]

    # Aplicar umbral optimizado
    prediction = int(probability >= threshold)

    return {
        "model": model_name,
        "prediction": prediction,
        "quality_label": target_labels[prediction],
        "probability_high_quality": round(float(probability), 4),
        "threshold_used": round(float(threshold), 4)
    }
