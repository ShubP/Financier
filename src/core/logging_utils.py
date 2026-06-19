"""Minimal shared logging setup."""

from __future__ import annotations

import logging

from .config import load_config

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring handlers once."""
    global _CONFIGURED
    if not _CONFIGURED:
        level_name = load_config().get("logging.level", "INFO")
        logging.basicConfig(
            level=getattr(
                logging, str(level_name).upper(), logging.INFO
            ),
            format=(
                "%(asctime)s | %(levelname)-7s | %(name)s | "
                "%(message)s"
            ),
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
