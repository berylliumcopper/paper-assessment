"""Load API settings for assessment workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_API_SETTINGS_PATH = Path(".secrets/assessment_api.json")


def load_api_settings(config_path: Path | None = None) -> dict:
    """Load API settings from file + environment.

    Precedence:
    1) Environment variables
    2) Local config file
    3) Built-in defaults
    """
    settings = {
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-1.5-flash",
    }

    path = config_path or DEFAULT_API_SETTINGS_PATH
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in ("api_key", "base_url", "model"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        settings[key] = value.strip()
        except Exception:  # noqa: BLE001
            pass

    # Gemini-first environment overrides, with OpenAI-compatible fallbacks.
    env_api_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    env_base_url = (
        os.getenv("GEMINI_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )
    env_model = (
        os.getenv("GEMINI_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
    )
    if env_api_key:
        settings["api_key"] = env_api_key
    if env_base_url:
        settings["base_url"] = env_base_url
    if env_model:
        settings["model"] = env_model

    return settings

