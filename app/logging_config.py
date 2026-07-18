import json
import logging
from typing import Any


logger = logging.getLogger("wine_api")
logger.setLevel(logging.INFO)


if not logger.handlers:
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(message)s"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def log_event(
    event_name: str,
    data: dict[str, Any],
    level: str = "info"
) -> None:
    """
    Registra un evento en formato JSON.

    event_name:
        Nombre del evento registrado.

    data:
        Información asociada al evento.

    level:
        Nivel del log: info, warning o error.
    """

    log_record = {
        "event": event_name,
        **data
    }

    message = json.dumps(
        log_record,
        ensure_ascii=False,
        default=str
    )

    if level == "error":
        logger.error(message)

    elif level == "warning":
        logger.warning(message)

    else:
        logger.info(message)