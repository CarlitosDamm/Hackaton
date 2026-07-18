from threading import Lock
from typing import Optional


request_count = 0
error_count = 0

latencies_ms: list[float] = []

prediction_count_by_class: dict[int, int] = {
    0: 0,
    1: 0
}

metrics_lock = Lock()


def update_metrics(
    status: str,
    latency_ms: Optional[float],
    prediction: Optional[int]
) -> None:
    """
    Actualiza las métricas globales después de procesar una solicitud.
    """

    global request_count
    global error_count

    with metrics_lock:
        request_count += 1

        if status == "error":
            error_count += 1

        if latency_ms is not None:
            latencies_ms.append(latency_ms)

        if prediction is not None:
            prediction_count_by_class[prediction] = (
                prediction_count_by_class.get(prediction, 0) + 1
            )


def get_metrics() -> dict:
    """
    Devuelve un resumen de las métricas acumuladas.
    """

    with metrics_lock:
        average_latency_ms = (
            sum(latencies_ms) / len(latencies_ms)
            if latencies_ms
            else 0.0
        )

        maximum_latency_ms = (
            max(latencies_ms)
            if latencies_ms
            else 0.0
        )

        error_rate = (
            error_count / request_count
            if request_count > 0
            else 0.0
        )

        return {
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": round(error_rate, 4),
            "average_latency_ms": round(
                average_latency_ms,
                2
            ),
            "maximum_latency_ms": round(
                maximum_latency_ms,
                2
            ),
            "prediction_count_by_class": (
                prediction_count_by_class.copy()
            )
        }