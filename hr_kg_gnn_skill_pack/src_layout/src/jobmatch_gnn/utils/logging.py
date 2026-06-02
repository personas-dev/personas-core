"""Logging utilities."""
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return configured logger."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
