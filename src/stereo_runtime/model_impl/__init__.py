"""Bundled model implementation code used by stereo runtime providers."""

from __future__ import annotations

import importlib
import sys


_ALIAS_PACKAGES = ("depth_anything_3", "InfiniDepth", "video_depth_anything")


for _name in _ALIAS_PACKAGES:
    _target = f"{__name__}.{_name}"
    if _name not in sys.modules:
        sys.modules[_name] = importlib.import_module(_target)

