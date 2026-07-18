import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent / "model" / "clasificador_vinos.joblib"

package = joblib.load(MODEL_PATH)

model = package["model"]
model_name = package["model_name"]
threshold = package["threshold"]
feature_names = package["feature_names"]
target_labels = package["target_labels"]


def predict_wine_quality(input_data: dict) -> dict:

    input_df = pd.DataFrame([{
        "fixed acidity": input_data["fixed_acidity"],
        "volatile acidity": input_data["volatile_acidity"],
        "citric acid": input_data["citric_acid"],
        "residual sugar": input_data["residual_sugar"],
        "chlorides": input_data["chlorides"],
        "free sulfur dioxide": input_data["free_sulfur_dioxide"],
        "total sulfur dioxide": input_data["total_sulfur_dioxide"],
        "density": input_data["density"],
        "pH": input_data["pH"],
        "sulphates": input_data["sulphates"],
        "alcohol": input_data["alcohol"],
        "wine_type": input_data["wine_type"]
    }])

    input_df = input_df[feature_names]

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= threshold)

    return {
        "model": model_name,
        "prediction": prediction,
        "quality_label": target_labels[prediction],
        "probability_high_quality": round(float(probability), 4),
        "threshold_used": round(float(threshold), 4)
    }