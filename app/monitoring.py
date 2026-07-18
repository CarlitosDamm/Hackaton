import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from app.logging_config import log_event
from app.metrics import update_metrics, get_metrics
from app.mlflow_tracking import log_monitoring_metrics
from app.alerts import evaluate_alerts
from app.drift import evaluate_input_drift


@dataclass
class RequestMonitoring:
    request_id: str
    start_time: float
    input_data: Optional[dict[str, Any]] = None
    latency_ms: Optional[float] = None
    prediction: Optional[int] = None
    status: str = "processing"
    error: Optional[str] = None


def start_request(
    input_data: Optional[dict[str, Any]] = None
) -> RequestMonitoring:

    return RequestMonitoring(
        request_id=str(uuid.uuid4()),
        start_time=time.perf_counter(),
        input_data=input_data
    )


def finish_request(
    request_monitoring: RequestMonitoring,
    prediction: Any
) -> RequestMonitoring:
    request_monitoring.latency_ms = (
        time.perf_counter() - request_monitoring.start_time
    ) * 1000

    request_monitoring.prediction = prediction
    request_monitoring.status = "success"

    return request_monitoring


def fail_request(
    request_monitoring: RequestMonitoring,
    error: Exception
) -> RequestMonitoring:
    request_monitoring.latency_ms = (
        time.perf_counter() - request_monitoring.start_time
    ) * 1000

    request_monitoring.status = "error"
    request_monitoring.error = str(error)

    return request_monitoring

def process_request(
    request_monitoring: RequestMonitoring
) -> None:

    log_data = {
        "request_id": request_monitoring.request_id,
        "status": request_monitoring.status,
        "latency_ms": request_monitoring.latency_ms,
        "prediction": request_monitoring.prediction,
        "error": request_monitoring.error
    }

    if request_monitoring.status == "error":
        log_event(
            event_name="prediction_request",
            data=log_data,
            level="error"
        )

    else:
        log_event(
            event_name="prediction_request",
            data=log_data,
            level="info"
        )

    update_metrics(
        status=request_monitoring.status,
        latency_ms=request_monitoring.latency_ms,
        prediction=request_monitoring.prediction
    )

    if request_monitoring.input_data is not None:
        evaluate_input_drift(
            request_monitoring.input_data
        )

    metrics_summary = get_metrics()

    evaluate_alerts(metrics_summary)

    log_monitoring_metrics(metrics_summary)