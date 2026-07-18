import os
from threading import Lock
from typing import Any, Optional

import pandas as pd

from app.logging_config import log_event

STANDARD_DEVIATION_LIMIT = 3.0

EXCLUDED_COLUMNS = {
    "quality",
    "target",
    "label",
    "prediction"
}


training_statistics: dict[str, dict[str, float]] = {}
drift_lock = Lock()
drift_enabled = False

drift_request_count = 0
drift_feature_count = 0

drift_count_by_feature: dict[str, int] = {}


def initialize_drift_detector() -> None:
    """
    Lee el dataset de entrenamiento y calcula las estadísticas
    necesarias para evaluar posibles casos de data drift.
    """

    global training_statistics
    global drift_count_by_feature
    global drift_enabled
    global drift_request_count
    global drift_feature_count

    try:

        RED_WINE_URL = (
            "https://archive.ics.uci.edu/ml/"
            "machine-learning-databases/"
            "wine-quality/"
            "winequality-red.csv"
        )

        WHITE_WINE_URL = (
            "https://archive.ics.uci.edu/ml/"
            "machine-learning-databases/"
            "wine-quality/"
            "winequality-white.csv"
        )

        red_wine = pd.read_csv(
            RED_WINE_URL,
            sep=";"
        )

        white_wine = pd.read_csv(
            WHITE_WINE_URL,
            sep=";"
        )

        red_wine["wine_type"] = 0
        white_wine["wine_type"] = 1

        training_data = pd.concat(
            [red_wine, white_wine],
            ignore_index=True
        )

        numeric_columns = training_data.select_dtypes(
            include="number"
        ).columns

        calculated_statistics: dict[
            str,
            dict[str, float]
        ] = {}

        for column in numeric_columns:

            if column.lower() in EXCLUDED_COLUMNS:
                continue

            mean_value = float(
                training_data[column].mean()
            )

            standard_deviation = float(
                training_data[column].std()
            )

            if pd.isna(standard_deviation):
                continue

            lower_limit = (
                mean_value
                - STANDARD_DEVIATION_LIMIT
                * standard_deviation
            )

            upper_limit = (
                mean_value
                + STANDARD_DEVIATION_LIMIT
                * standard_deviation
            )

            calculated_statistics[column] = {
                "mean": mean_value,
                "std": standard_deviation,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit
            }

        if not calculated_statistics:
            raise ValueError(
                "No se encontraron variables numéricas "
                "válidas en el dataset."
            )

        with drift_lock:
            training_statistics = calculated_statistics

            drift_count_by_feature = {
                feature: 0
                for feature in training_statistics
            }

            drift_request_count = 0
            drift_feature_count = 0
            drift_enabled = True

        log_event(
            event_name="drift_detector_initialized",
            data={
                "monitored_features": list(
                    training_statistics.keys()
                ),
                "standard_deviation_limit": (
                    STANDARD_DEVIATION_LIMIT
                )
            },
            level="info"
        )

    except Exception as error:
        drift_enabled = False

        log_event(
            event_name="drift_detector_initialization_failed",
            data={
                "error": str(error)
            },
            level="error"
        )


def evaluate_input_drift(
    input_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Compara una solicitud contra las estadísticas del dataset
    de entrenamiento.

    Devuelve las variables cuyos valores están fuera del rango
    media ± N desviaciones estándar.
    """

    global drift_request_count
    global drift_feature_count

    if not drift_enabled:
        return []

    detected_features: list[dict[str, Any]] = []

    with drift_lock:
        drift_request_count += 1

        for feature, statistics in training_statistics.items():

            if feature not in input_data:
                continue

            raw_value = input_data[feature]

            try:
                current_value = float(raw_value)
            except (TypeError, ValueError):
                continue

            lower_limit = statistics["lower_limit"]
            upper_limit = statistics["upper_limit"]

            if (
                current_value < lower_limit
                or current_value > upper_limit
            ):
                drift_feature_count += 1

                drift_count_by_feature[feature] = (
                    drift_count_by_feature.get(feature, 0)
                    + 1
                )

                detected_features.append({
                    "feature": feature,
                    "current_value": current_value,
                    "training_mean": round(
                        statistics["mean"],
                        4
                    ),
                    "training_std": round(
                        statistics["std"],
                        4
                    ),
                    "lower_limit": round(
                        lower_limit,
                        4
                    ),
                    "upper_limit": round(
                        upper_limit,
                        4
                    )
                })

    if detected_features:
        log_event(
            event_name="data_drift_detected",
            data={
                "drifted_feature_count": len(
                    detected_features
                ),
                "features": detected_features
            },
            level="warning"
        )

    return detected_features


def get_drift_status() -> dict[str, Any]:
    """
    Devuelve el estado acumulado del detector de drift.
    """

    with drift_lock:

        return {
            "enabled": drift_enabled,
            "standard_deviation_limit": (
                STANDARD_DEVIATION_LIMIT
            ),
            "monitored_feature_count": len(
                training_statistics
            ),
            "evaluated_request_count": drift_request_count,
            "total_drifted_feature_events": (
                drift_feature_count
            ),
            "drift_count_by_feature": (
                drift_count_by_feature.copy()
            ),
            "training_statistics": {
                feature: {
                    key: round(value, 4)
                    for key, value in statistics.items()
                }
                for feature, statistics
                in training_statistics.items()
            }
        }