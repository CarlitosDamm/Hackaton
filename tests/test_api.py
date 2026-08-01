from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type": 0
}


def test_health_endpoint() -> None:
    """La API debe reportar que está disponible y que el modelo está cargado."""
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert isinstance(body["model_name"], str)
    assert body["model_name"]


def test_predict_with_valid_data() -> None:
    """Una entrada válida debe generar una predicción estructurada."""
    response = client.post("/predict", json=VALID_WINE)

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in (0, 1)
    assert body["quality_label"] in ("Estándar", "Alta calidad")
    assert 0.0 <= body["probability_high_quality"] <= 1.0
    assert 0.0 <= body["threshold_used"] <= 1.0
    assert isinstance(body["model"], str)


def test_predict_rejects_incomplete_data() -> None:
    """Pydantic debe rechazar solicitudes que omitan características requeridas."""
    incomplete_wine = {
        "fixed_acidity": 7.4,
        "alcohol": 9.4
    }

    response = client.post("/predict", json=incomplete_wine)

    assert response.status_code == 422