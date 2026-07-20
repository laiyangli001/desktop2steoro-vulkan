"""Assemble the Python capture and stereo pipeline for the Vulkan project."""

from __future__ import annotations

import os
import platform
import threading
import time
from pathlib import Path

from capture import capture_frame_to_rgb, prepare_rgb_for_stereo_runtime
from capture.session import CaptureSessionLoop
from stereo_runtime.pipeline import RuntimePipelineLoop
from utils import (
    CAPTURE_MODE,
    CAPTURE_TOOL,
    CONVERGENCE,
    DEPTH_STRENGTH,
    DEVICE,
    DEVICE_INFO,
    FPS,
    MONITOR_INDEX,
    OS_NAME,
    OUTPUT_RESOLUTION,
    RENDER_SIZE_CONFIG,
    RUN_MODE,
    WINDOW_TITLE,
    _get_settings,
    shutdown_event,
)

from .runtime_callbacks import RuntimeCallbacks
from .runtime_context import (
    build_capture_callbacks,
    build_runtime_pipeline_context,
    create_runtime_context,
)
from .runtime_output import CudaVulkanOutputAdapter, VulkanRuntimeOutputConsumer


def _openxr_filament_config(settings: dict) -> dict[str, object]:
    """Resolve the packaged Windows Filament scene for direct OpenXR runs."""
    src_root = Path(__file__).resolve().parents[1]
    platform_bridge = {
        "Windows": src_root / "xr_viewer" / "native" / "filament_bridge.dll",
        "Linux": src_root / "xr_viewer" / "native" / "libfilament_bridge.so",
        "Darwin": src_root / "xr_viewer" / "native" / "libfilament_bridge.dylib",
    }.get(platform.system())
    glb_path = src_root / "xr_viewer" / "environments" / "Artemis" / "environment.glb"
    profile_path = src_root / "xr_viewer" / "environments" / "Artemis" / "profile.json"

    bridge_path = os.environ.get("D2S_FILAMENT_BRIDGE") or (
        str(platform_bridge) if platform_bridge and platform_bridge.is_file() else None
    )
    configured_glb = os.environ.get("D2S_FILAMENT_GLB")
    configured_profile = os.environ.get("D2S_FILAMENT_PROFILE")
    return {
        "swapchain_color_mode": str(
            settings.get("OpenXR Color Mode", "sRGB")
        ).strip().lower(),
        "filament_bridge_path": bridge_path,
        "filament_glb_path": configured_glb or (str(glb_path) if glb_path.is_file() else None),
        "filament_profile_path": configured_profile
        or (str(profile_path) if profile_path.is_file() else None),
        "filament_scene_exposure_ev": float(
            settings.get("Filament Scene Exposure", 2.0)
        ),
        "filament_skybox_brightness": float(
            settings.get("Filament Skybox Brightness", 1.0)
        ),
    }


def _queue_clear(queue) -> None:
    while True:
        try:
            queue.get_nowait()
        except Exception:
            return


def run_processing_runtime(*, max_seconds: float | None = None) -> int:
    """Run capture, inference, and pipeline threads until shutdown is requested."""

    shutdown_event.clear()
    settings = _get_settings()
    context = create_runtime_context(
        file_path=str(Path(__file__).resolve().parents[1] / "main.py"),
        settings=settings,
        cache_path=str(Path(__file__).resolve().parents[1] / "models"),
        device=DEVICE,
        device_info=DEVICE_INFO,
        output_resolution=OUTPUT_RESOLUTION,
        render_size_config=RENDER_SIZE_CONFIG,
        fps=FPS,
        window_title=WINDOW_TITLE,
        capture_mode=CAPTURE_MODE,
        monitor_index=MONITOR_INDEX,
        capture_tool=CAPTURE_TOOL,
        os_name=OS_NAME,
        run_mode=RUN_MODE,
        depth_strength=DEPTH_STRENGTH,
        convergence=CONVERGENCE,
    )
    callbacks = RuntimeCallbacks(context)

    capture_callbacks = build_capture_callbacks(
        raw_q=context.raw_q,
        shutdown_event=shutdown_event,
        queue_clear=callbacks.queue_clear_nonblocking,
        inc_source_stat=callbacks.source_stat_inc,
        inc_breakdown=callbacks.breakdown_inc,
        put_raw_latest=callbacks.put_raw_latest,
        is_paused=callbacks.openxr_source_paused,
        is_hard_idle=callbacks.openxr_hard_idle_active,
        on_session_update=callbacks.capture_session_update,
        on_tick=callbacks.log_source_health,
    )

    pipeline_context = build_runtime_pipeline_context(
        shutdown_event=shutdown_event,
        app_context=context,
        run_mode=RUN_MODE,
        device=DEVICE,
        capture_frame_to_rgb=capture_frame_to_rgb,
        prepare_rgb_for_stereo_runtime=prepare_rgb_for_stereo_runtime,
        current_openxr_render_config=callbacks.current_openxr_render_config,
        is_hard_idle=callbacks.openxr_hard_idle_active,
        is_source_paused=callbacks.openxr_source_paused,
        log_source_health=callbacks.log_source_health,
        source_stat_inc=callbacks.source_stat_inc,
        breakdown_inc=callbacks.breakdown_inc,
        breakdown_add_time=callbacks.breakdown_add_time,
        breakdown_add_runtime_timing=callbacks.breakdown_add_runtime_timing,
        set_preprocess_backend=callbacks.set_runtime_preprocess_backend,
        queue_clear=callbacks.queue_clear_nonblocking,
        queue_drain_latest=callbacks.queue_drain_latest,
        queue_put_latest=callbacks.queue_put_latest,
        log_stereo_runtime_mode_once=callbacks.log_stereo_runtime_mode_once,
        apply_stereo_hot_reload_if_needed=callbacks.apply_stereo_hot_reload_if_needed,
        warmup_stereo_once_for_frame=callbacks.warmup_stereo_once_for_frame,
        log_fast_plus_fused_runtime_state=callbacks.log_fast_plus_fused_runtime_state,
    )

    capture_thread = threading.Thread(
        target=CaptureSessionLoop(context.capture_config, capture_callbacks).run,
        args=(shutdown_event,),
        name="VulkanCapture",
        daemon=True,
    )
    pipeline = RuntimePipelineLoop(pipeline_context)
    pipeline_thread = threading.Thread(
        target=pipeline.run,
        name="VulkanStereoPipeline",
        daemon=True,
    )
    presenter = None
    presenter_thread = None
    output_consumer = None
    output_thread = None
    if str(RUN_MODE).strip().lower() == "openxr":
        from xr_viewer.core_openxr_vulkan import OpenXrVulkanConfig, OpenXrVulkanPresenter

        filament_config = _openxr_filament_config(settings)
        presenter = OpenXrVulkanPresenter(
            OpenXrVulkanConfig(**filament_config)
        )
        presenter_thread = threading.Thread(
            target=presenter.run_until,
            args=(shutdown_event,),
            name="VulkanOpenXRPresenter",
            daemon=True,
        )
        presenter_thread.start()
        output_consumer = VulkanRuntimeOutputConsumer(
            runtime_q=context.runtime_q,
            shutdown_event=shutdown_event,
            source_stat_inc=callbacks.source_stat_inc,
            sink=presenter,
            gpu_adapter=CudaVulkanOutputAdapter(presenter),
        )
        output_thread = threading.Thread(
            target=output_consumer.run,
            name="VulkanOutputConsumer",
            daemon=True,
        )
        output_thread.start()
    capture_thread.start()
    pipeline_thread.start()
    print(
        f"Desktop2Stereo Vulkan runtime started: mode={RUN_MODE} device={DEVICE_INFO}",
        flush=True,
    )

    deadline = None if max_seconds is None else time.monotonic() + max(0.0, max_seconds)
    try:
        while not shutdown_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        callbacks.stop_active_capture_session()
        _queue_clear(context.raw_q)
        _queue_clear(context.runtime_q)
        pipeline_thread.join(timeout=2.0)
        capture_thread.join(timeout=2.0)
        if output_thread is not None:
            output_thread.join(timeout=2.0)
        if output_consumer is not None:
            output_consumer.close()
        if presenter_thread is not None:
            presenter_thread.join(timeout=2.0)
        if presenter is not None:
            presenter.close()
        close = getattr(context.stereo_runtime, "close", None)
        if callable(close):
            close()
    return 0
