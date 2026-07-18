from __future__ import annotations

from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider


def create_platform_depth_provider(config: DepthProviderConfig | dict | None = None):
    """Compatibility factory for the future platform provider layout."""
    return create_depth_provider(config)

