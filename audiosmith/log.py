"""Minimal logging setup for AudioSmith — stdlib only."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Configure the 'audiosmith' root logger with console and optional file output."""
    logger = logging.getLogger('audiosmith')
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.setLevel(level.upper())
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'audiosmith' namespace."""
    return logging.getLogger(f'audiosmith.{name}')
