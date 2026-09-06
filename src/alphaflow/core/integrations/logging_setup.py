"""setup_logger indirection.

The spec says logging comes from an existing shared package rather than being built here.
This module prefers that package when importable and otherwise configures an equivalent
console + rotating-file logger, so no caller has to care which one it got.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_PATH_ENV_VAR = "ALPHAFLOW_LOG_PATH"
MAX_BYTES = 20 * 1024 * 1024
BACKUP_COUNT = 5


def _external_setup_logger() -> Any | None:
    try:
        from apcr_desktool.logging import setup_logger
        return setup_logger
    except ImportError:
        return None


def setup_logger(name: str, level: int = logging.INFO, log_path: str | Path | None = None) -> logging.Logger:
    """Idempotent - repeated calls for the same name will not stack handlers."""
    external = _external_setup_logger()
    if external is not None:
        return external(name, level)

    logger = logging.getLogger(name)
    if getattr(logger, "_alphaflow_configured", False):
        return logger
    logger.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)

    resolved_path = log_path or os.environ.get(LOG_PATH_ENV_VAR)
    if resolved_path:
        directory = Path(resolved_path)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(directory / "alphaflow.log", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    logger._alphaflow_configured = True
    return logger
