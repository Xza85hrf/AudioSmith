"""Minimal logging setup for AudioSmith — stdlib only."""

import logging


def setup_logging(level='INFO', log_file=None):
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


def get_logger(name):
    """Return a child logger under the 'audiosmith' namespace."""
    return logging.getLogger(f'audiosmith.{name}')
