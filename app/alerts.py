from threading import Lock
from typing import Any

from app.logging_config import log_event


MIN_REQUESTS_FOR_ALERT = 5
MAX_ERROR_RATE = 0.20
MAX_AVERAGE_LATENCY_MS = 500.0
MAX_CLASS_PROPORTION = 0.90

active_alerts = {
    "high_error_rate": False,
    "high_average_latency": False,
    "prediction_distribution_imbalance": False
}

alerts_lock = Lock()

def activate_alert(
        alert_type: str,
        alert_data: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Activa una alerta unicamente si no está activa.
    """

    with alerts_lock:
        if active_alerts[alert_type]:
            return None
        
        active_alerts[alert_type] = True
    
    alert = {
        "alert_type": alert_type,
        "status": "active",
        **alert_data
    }

    log_event(
        event_name="monitoring_alert",
        data=alert,
        level="warning"
    )

    return alert


def recover_alert(
    alert_type: str,
    recovery_data: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Genera un evento de recuperación únicamente
    cuando una alerta estaba previamente activa.
    """

    with alerts_lock:
        if not active_alerts[alert_type]:
            return None

        active_alerts[alert_type] = False

    recovery = {
        "alert_type": alert_type,
        "status": "recovered",
        **recovery_data
    }

    log_event(
        event_name="monitoring_alert_recovered",
        data=recovery,
        level="info"
    )

    return recovery


def evaluate_alerts(
    metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Evalúa las métricas acumuladas.

    Genera alertas solamente cuando una condición cambia
    de estado normal a estado de alerta.

    También genera un evento cuando la condición se recupera.
    """

    generated_events: list[dict[str, Any]] = []

    request_count = metrics["request_count"]

    if request_count < MIN_REQUESTS_FOR_ALERT:
        return generated_events

    error_rate = metrics["error_rate"]
    average_latency_ms = metrics["average_latency_ms"]

    prediction_counts = metrics[
        "prediction_count_by_class"
    ]

    total_predictions = sum(
        prediction_counts.values()
    )

    # Alerta por tasa de error
    if error_rate > MAX_ERROR_RATE:
        alert = activate_alert(
            alert_type="high_error_rate",
            alert_data={
                "current_value": error_rate,
                "threshold": MAX_ERROR_RATE
            }
        )

        if alert is not None:
            generated_events.append(alert)

    else:
        recovery = recover_alert(
            alert_type="high_error_rate",
            recovery_data={
                "current_value": error_rate,
                "threshold": MAX_ERROR_RATE
            }
        )

        if recovery is not None:
            generated_events.append(recovery)

    # Alerta por latencia promedio
    if average_latency_ms > MAX_AVERAGE_LATENCY_MS:
        alert = activate_alert(
            alert_type="high_average_latency",
            alert_data={
                "current_value": average_latency_ms,
                "threshold": MAX_AVERAGE_LATENCY_MS
            }
        )

        if alert is not None:
            generated_events.append(alert)

    else:
        recovery = recover_alert(
            alert_type="high_average_latency",
            recovery_data={
                "current_value": average_latency_ms,
                "threshold": MAX_AVERAGE_LATENCY_MS
            }
        )

        if recovery is not None:
            generated_events.append(recovery)

    # Alerta por distribución de predicciones
    imbalance_detected = False
    dominant_class = None
    dominant_proportion = 0.0

    if total_predictions > 0:
        for prediction_class, count in prediction_counts.items():

            class_proportion = count / total_predictions

            if class_proportion > dominant_proportion:
                dominant_proportion = class_proportion
                dominant_class = prediction_class

            if class_proportion > MAX_CLASS_PROPORTION:
                imbalance_detected = True

    if imbalance_detected:
        alert = activate_alert(
            alert_type="prediction_distribution_imbalance",
            alert_data={
                "prediction_class": dominant_class,
                "current_value": round(
                    dominant_proportion,
                    4
                ),
                "threshold": MAX_CLASS_PROPORTION
            }
        )

        if alert is not None:
            generated_events.append(alert)

    else:
        recovery = recover_alert(
            alert_type="prediction_distribution_imbalance",
            recovery_data={
                "current_value": round(
                    dominant_proportion,
                    4
                ),
                "threshold": MAX_CLASS_PROPORTION
            }
        )

        if recovery is not None:
            generated_events.append(recovery)

    return generated_events


def get_alert_status() -> dict[str, bool]:
    """
    Devuelve una copia del estado actual de las alertas.
    """

    with alerts_lock:
        return active_alerts.copy()