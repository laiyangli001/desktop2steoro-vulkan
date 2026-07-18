from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunModeConfig:
    run_mode: str
    stream_mode: str | None
    use_3d_monitor: bool
    lossless_scaling_support: bool
    fix_viewer_aspect: bool


def resolve_run_mode(
    raw_run_mode: str,
    *,
    os_name: str,
    fix_viewer_aspect: bool,
    lossless_scaling_support: bool,
) -> RunModeConfig:
    use_3d_monitor = False
    stream_mode = None
    resolved_lossless_scaling = False
    resolved_fix_viewer_aspect = True if raw_run_mode == "RTMP Streamer" else fix_viewer_aspect

    if raw_run_mode == "Local Viewer":
        run_mode = "Viewer"
    elif raw_run_mode == "3D Monitor" and os_name == "Windows":
        run_mode = "Viewer"
        use_3d_monitor = True
    elif raw_run_mode == "MJPEG Streamer":
        run_mode = "Viewer"
        stream_mode = "MJPEG"
    elif raw_run_mode == "RTMP Streamer":
        run_mode = "Viewer"
        stream_mode = "RTMP"
        if os_name == "Windows":
            resolved_lossless_scaling = lossless_scaling_support
    elif raw_run_mode == "OpenXR Link":
        run_mode = "OpenXR"
    else:
        run_mode = "Streamer"

    return RunModeConfig(
        run_mode=run_mode,
        stream_mode=stream_mode,
        use_3d_monitor=use_3d_monitor,
        lossless_scaling_support=resolved_lossless_scaling,
        fix_viewer_aspect=resolved_fix_viewer_aspect,
    )
