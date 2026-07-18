import os

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

from threading import Lock
from typing import Any, Optional

from mlflow import MlflowClient
from mlflow.entities import RunStatus

from app.logging_config import log_event


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "wine-quality-api-monitoring"
)


mlflow_client: Optional[MlflowClient] = None
mlflow_run_id: Optional[str] = None
mlflow_step = 0
mlflow_enabled = False

mlflow_lock = Lock()


def initialize_mlflow(
    model_name: str
) -> None:
    """
    Inicializa la conexión con MLflow y crea un run
    para la sesión actual de FastAPI.

    Si MLflow no está disponible, registra el error,
    pero no detiene la API.
    """

    global mlflow_client
    global mlflow_run_id
    global mlflow_step
    global mlflow_enabled

    try:
        client = MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI
        )

        experiment = client.get_experiment_by_name(
            MLFLOW_EXPERIMENT_NAME
        )

        if experiment is None:
            experiment_id = client.create_experiment(
                MLFLOW_EXPERIMENT_NAME
            )
        else:
            experiment_id = experiment.experiment_id

        run = client.create_run(
            experiment_id=experiment_id,
            tags={
                "mlflow.runName": "api-monitoring-session",
                "model_name": model_name,
                "environment": "development",
                "endpoint": "/predict",
                "monitoring_type": "inference"
            }
        )

        mlflow_client = client
        mlflow_run_id = run.info.run_id
        mlflow_step = 0
        mlflow_enabled = True

        log_event(
            event_name="mlflow_initialized",
            data={
                "tracking_uri": MLFLOW_TRACKING_URI,
                "experiment_name": MLFLOW_EXPERIMENT_NAME,
                "run_id": mlflow_run_id
            },
            level="info"
        )

    except Exception as error:
        mlflow_client = None
        mlflow_run_id = None
        mlflow_enabled = False

        log_event(
            event_name="mlflow_initialization_failed",
            data={
                "tracking_uri": MLFLOW_TRACKING_URI,
                "error": str(error)
            },
            level="error"
        )


def log_monitoring_metrics(
    metrics: dict[str, Any]
) -> None:
    """
    Registra en MLflow el resumen actualizado
    de las métricas de monitoreo.
    """

    global mlflow_step
    global mlflow_enabled

    if (
        not mlflow_enabled
        or mlflow_client is None
        or mlflow_run_id is None
    ):
        return

    try:
        with mlflow_lock:
            mlflow_step += 1
            current_step = mlflow_step

            prediction_counts = metrics.get(
                "prediction_count_by_class",
                {}
            )

            metrics_to_log = {
                "request_count": float(
                    metrics["request_count"]
                ),
                "error_count": float(
                    metrics["error_count"]
                ),
                "error_rate": float(
                    metrics["error_rate"]
                ),
                "average_latency_ms": float(
                    metrics["average_latency_ms"]
                ),
                "maximum_latency_ms": float(
                    metrics["maximum_latency_ms"]
                ),
                "prediction_class_0_count": float(
                    prediction_counts.get(0, 0)
                ),
                "prediction_class_1_count": float(
                    prediction_counts.get(1, 0)
                )
            }

            for metric_name, metric_value in metrics_to_log.items():
                mlflow_client.log_metric(
                    run_id=mlflow_run_id,
                    key=metric_name,
                    value=metric_value,
                    step=current_step
                )

    except Exception as error:
        mlflow_enabled = False

        log_event(
            event_name="mlflow_metric_logging_failed",
            data={
                "run_id": mlflow_run_id,
                "error": str(error)
            },
            level="error"
        )


def close_mlflow_run() -> None:
    """
    Marca como terminado el run de monitoreo
    cuando FastAPI se apaga.
    """

    global mlflow_enabled

    if (
        mlflow_client is None
        or mlflow_run_id is None
    ):
        return

    try:
        mlflow_client.set_terminated(
            run_id=mlflow_run_id,
            status=RunStatus.to_string(
                RunStatus.FINISHED
            )
        )

        log_event(
            event_name="mlflow_run_closed",
            data={
                "run_id": mlflow_run_id
            },
            level="info"
        )

    except Exception as error:
        log_event(
            event_name="mlflow_run_close_failed",
            data={
                "run_id": mlflow_run_id,
                "error": str(error)
            },
            level="error"
        )

    finally:
        mlflow_enabled = False