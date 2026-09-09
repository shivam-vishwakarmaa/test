"""Typed access to config/config.yaml with environment overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

DEFAULT_PATH = os.environ.get("SA_CONFIG", "config/config.yaml")


class Config(dict):
    """dict with dotted-path access: cfg.get_path('llm.backend')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


# Environment variables that override config values, for CI and quick sweeps.
_ENV_OVERRIDES = {
    "LLM_BACKEND": "llm.backend",
    "EMBED_BACKEND": "embedding.backend",
    "AGENT_MODEL": "agent.model",
    "JUDGE_MODEL": "judge.model",
    "BRAND": "brand",
}


def _set_path(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


@lru_cache(maxsize=8)
def load_config(path: str = DEFAULT_PATH) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for env_key, dotted in _ENV_OVERRIDES.items():
        val = os.environ.get(env_key)
        if val:
            _set_path(raw, dotted, val)
    return Config(raw)
