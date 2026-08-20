"""Centralized logging infrastructure for FedTROS-PR experiments."""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class CentralFormatter(logging.Formatter):
    """Standardized single-line log formatter with precise timestamps."""

    def __init__(self) -> None:
        super().__init__(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)


def configure_logging(
    run_dir: Path | str | None = None,
    *,
    console_level: int | str = logging.INFO,
    file_level: int | str = logging.DEBUG,
    capture_third_party: bool = True,
) -> Path | None:
    """Configure centralized logging for console and file output.

    Automatically creates outputs/runs/<run_id>/logs/run.log and debug.log if run_dir is provided.
    """
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper(), logging.INFO)
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper(), logging.DEBUG)

    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": console_level,
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    }
    root_handlers = ["console"]
    log_file_path: Path | None = None

    if run_dir is not None:
        run_path = Path(run_dir)
        log_dir = run_path / "logs" if run_path.name != "logs" else run_path
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "run.log"
        debug_file_path = log_dir / "debug.log"

        handlers["run_file"] = {
            "class": "logging.FileHandler",
            "level": console_level,
            "formatter": "standard",
            "filename": str(log_file_path),
            "encoding": "utf-8",
        }
        handlers["debug_file"] = {
            "class": "logging.FileHandler",
            "level": file_level,
            "formatter": "standard",
            "filename": str(debug_file_path),
            "encoding": "utf-8",
        }
        root_handlers.extend(["run_file", "debug_file"])

    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": LOG_FORMAT,
                "datefmt": DATE_FORMAT,
            }
        },
        "handlers": handlers,
        "root": {
            "handlers": root_handlers,
            "level": min(console_level, file_level),
        },
        "loggers": {
            "flwr": {"level": "WARNING" if not capture_third_party else "INFO", "propagate": True},
            "PIL": {"level": "WARNING", "propagate": True},
            "urllib3": {"level": "WARNING", "propagate": True},
            "wandb": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(config_dict)
    return log_file_path


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured under the centralized logging hierarchy."""
    return logging.getLogger(name)
