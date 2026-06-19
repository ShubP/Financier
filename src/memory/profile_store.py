"""Persistent user profiles (JSON file, optional DynamoDB).

Stores per-user preferences: name, risk profile, holdings, and a
savings goal. Conversation history lives in the UI session, not here.
The DynamoDB backend is optional and degrades to JSON if boto3 or AWS
credentials are unavailable.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import Any

from ..core.config import Config, load_config
from ..core.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_PROFILE: dict[str, Any] = {
    "name": "",
    "risk_profile": "moderate",
    "holdings": [],
    "goal": {},
}


def _merge_defaults(profile: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_PROFILE)
    merged.update(profile or {})
    return merged


class ProfileStore:
    """Load/save user profiles with a pluggable backend."""

    def __init__(self, cfg: Config | None = None) -> None:
        cfg = cfg or load_config()
        self._backend = str(cfg.get("memory.backend", "json"))
        self._path = cfg.path(
            "memory.profiles_path", "data/user_profiles.json"
        )
        self._lock = Lock()
        self._dynamo: Any = None
        if self._backend == "dynamodb":
            self._init_dynamo(cfg)

    def get(self, user_id: str) -> dict[str, Any]:
        if self._dynamo is not None:
            return self._dynamo_get(user_id)
        return _merge_defaults(self._read_all().get(user_id))

    def save(self, user_id: str, profile: dict[str, Any]) -> None:
        if self._dynamo is not None:
            self._dynamo_put(user_id, profile)
            return
        with self._lock:
            data = self._read_all()
            data[user_id] = profile
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )

    # --- JSON backend ---
    def _read_all(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read profiles: %s", exc)
            return {}

    # --- optional DynamoDB backend ---
    def _init_dynamo(self, cfg: Config) -> None:
        try:
            import boto3

            table = cfg.get("memory.dynamodb.table", "financier-profiles")
            region = cfg.get("memory.dynamodb.region", "us-east-1")
            self._dynamo = boto3.resource(
                "dynamodb", region_name=region
            ).Table(table)
            logger.info("Profile store: DynamoDB table %s", table)
        except Exception as exc:
            logger.warning("DynamoDB unavailable, using JSON: %s", exc)
            self._dynamo = None

    def _dynamo_get(self, user_id: str) -> dict[str, Any]:
        try:
            item = self._dynamo.get_item(
                Key={"user_id": user_id}
            ).get("Item")
            if not item:
                return dict(_DEFAULT_PROFILE)
            return _merge_defaults(json.loads(item["profile"]))
        except Exception as exc:
            logger.warning("DynamoDB get failed: %s", exc)
            return dict(_DEFAULT_PROFILE)

    def _dynamo_put(self, user_id: str, profile: dict[str, Any]) -> None:
        try:
            self._dynamo.put_item(
                Item={
                    "user_id": user_id,
                    "profile": json.dumps(profile),
                }
            )
        except Exception as exc:
            logger.warning("DynamoDB put failed: %s", exc)


_INSTANCE: ProfileStore | None = None


def get_profile_store(cfg: Config | None = None) -> ProfileStore:
    """Return a process-wide :class:`ProfileStore` singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ProfileStore(cfg)
    return _INSTANCE
