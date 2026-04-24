"""
Logging configuration for the baby monitor pipeline.

Call setup_logging() once at startup (in main.py) to configure all loggers.
All modules use `logger = logging.getLogger(__name__)` for structured output.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure the root logger with a consistent format.

    Args:
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to also write logs to a file.
    """
    fmt = "%(asctime)s [%(levelname)-7s] %(name)-30s │ %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # override any prior basicConfig
    )

    # Quieten noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
