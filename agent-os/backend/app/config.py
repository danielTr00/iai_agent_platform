"""Laedt config.yaml + relevante Env-Variablen."""
import os
import yaml

_CONFIG = None


def load() -> dict:
    global _CONFIG
    if _CONFIG is None:
        path = os.environ.get("CONFIG_PATH", "config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
