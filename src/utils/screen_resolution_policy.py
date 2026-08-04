from __future__ import annotations

from dataclasses import dataclass


# These are policy buckets for filtering, not forced image dimensions. The
# source frame keeps its real aspect ratio and actual width/height.
STANDARD_INPUT_RESOLUTIONS = {
    1: (1920, 1080),
    2: (2560, 1440),
    4: (3840, 2160),
}
INPUT_TO_HEADSET_TIER = {1: 2, 2: 4, 4: 8}
HEADSET_TIERS = (2, 4, 8)


@dataclass(frozen=True, slots=True)
class ScreenSamplingPlan:
    source_width: int
    source_height: int
    input_tier_k: int
    headset_tier_k: int
    recommended_headset_tier_k: int
    effective_tier_k: int
    filter_scale: float
    upscale_scale: float
    mode: str

    @property
    def source_texel_size(self) -> tuple[float, float]:
        return (1.0 / self.source_width, 1.0 / self.source_height)

    @property
    def matrix_label(self) -> str:
        return (
            f"input_{self.input_tier_k}k->headset_{self.recommended_headset_tier_k}k "
            f"selected={self.headset_tier_k}k effective={self.effective_tier_k}k"
        )


def classify_input_resolution(width: int, height: int) -> int:
    """Map arbitrary input geometry to the nearest supported horizontal tier."""
    long_edge = max(int(width), int(height))
    if long_edge <= 0:
        raise ValueError("input resolution must be positive")
    if long_edge <= 1920:
        return 1
    if long_edge <= 2880:
        return 2
    return 4


def build_screen_sampling_plan(
    source_width: int,
    source_height: int,
    headset_tier_k: int,
) -> ScreenSamplingPlan:
    """Build the input/headset matrix without changing the source geometry.

    The selected headset tier is capped by the 2x input-to-headset policy. If
    a lower-tier headset receives a higher-tier source, the filter footprint is
    widened only enough to prefilter that downsample; matched paths stay at the
    source texel footprint and keep the existing sharp path.
    """
    width = int(source_width)
    height = int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("source resolution must be positive")
    input_tier = classify_input_resolution(width, height)
    headset_tier = int(headset_tier_k)
    if headset_tier not in HEADSET_TIERS:
        raise ValueError(f"unsupported headset resolution tier: {headset_tier_k}")
    recommended = INPUT_TO_HEADSET_TIER[input_tier]
    effective = min(headset_tier, recommended)
    filter_scale = max(1.0, input_tier / float(effective))
    upscale_scale = max(1.0, effective / float(input_tier))
    if upscale_scale > 1.0:
        mode = "upscale_easu"
    elif filter_scale > 1.0:
        mode = "downsample_lanczos_rcas"
    else:
        mode = "native_mip"
    return ScreenSamplingPlan(
        source_width=width,
        source_height=height,
        input_tier_k=input_tier,
        headset_tier_k=headset_tier,
        recommended_headset_tier_k=recommended,
        effective_tier_k=effective,
        filter_scale=filter_scale,
        upscale_scale=upscale_scale,
        mode=mode,
    )
