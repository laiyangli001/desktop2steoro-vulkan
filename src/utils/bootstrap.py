from __future__ import annotations

import os

from .network import configure_huggingface_endpoint
from .platform_env import configure_platform_environment
from .settings import load_settings


def bootstrap_settings(path: str, *, os_name: str) -> dict:
    settings = load_settings(path)
    configure_platform_environment(os_name)
    configure_huggingface_endpoint()
    if str(settings.get("Debug Mode", False) or False).strip().lower() in ("1", "true", "yes", "on"):
        os.environ["D2S_DEBUG"] = "1"
    return settings
