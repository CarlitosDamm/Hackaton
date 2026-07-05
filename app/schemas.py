from pydantic import BaseModel


class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    wine_type: float


class PredictionResponse(BaseModel):
    model: str
    prediction: int
    quality_label: str
    probability_high_quality: float
    threshold_used: float