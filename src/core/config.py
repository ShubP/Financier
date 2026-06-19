"""Configuration loading.

Reads ``config.yaml`` from the project root, merges environment
variables (via ``.env`` if present), and exposes a small
dot-accessible ``Config`` object. Secrets never live in YAML — they
always come from the environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is a dependency, but stay import-safe.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


# Project root: two levels up from src/core/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config:
    """Parsed config with ``get("a.b.c")`` dotted access."""

    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self._data = data
        self.root = root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested value via a dotted path."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted_key: str, default: str | None = None) -> Path:
        """Resolve a configured relative path against the root."""
        value = self.get(dotted_key, default)
        if value is None:
            raise KeyError(f"No path configured at '{dotted_key}'")
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    @property
    def data(self) -> dict[str, Any]:
        return self._data


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load configuration once and cache it for the process."""
    load_dotenv(PROJECT_ROOT / ".env")
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:  # pragma: no cover
        data = {}
    return Config(data, PROJECT_ROOT)


def env(name: str, default: str | None = None) -> str | None:
    """Read an env var, treating empty strings as unset."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()
