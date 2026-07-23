"""Assemble the Python capture and stereo pipeline for the Vulkan project."""

from __future__ import annotations

import json
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


def _resolve_filament_environment_paths(
    settings: dict,
    src_root: Path,
) -> tuple[Path | None, Path | None]:
    environment_name = str(settings.get("Environment Model", "")).strip()
    selected_name = (
        "Default"
        if not environment_name or environment_name.lower() == "none"
        else environment_name
    )
    environments_root = src_root / "xr_viewer" / "environments"

    def resolve(name: str) -> tuple[Path | None, Path]:
        room_dir = environments_root / name
        profile_path = room_dir / "profile.json"
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"OpenXR environment profile not found: {profile_path}"
            )
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"OpenXR environment profile is invalid: {profile_path}"
            ) from exc
        if not isinstance(profile, dict):
            raise ValueError(
                f"OpenXR environment profile root must be an object: {profile_path}"
            )

        glb_value = profile.get("glb", "environment.glb")
        if glb_value in (None, "", False):
            return None, profile_path
        glb_path = room_dir / str(glb_value)
        if not glb_path.is_file():
            raise FileNotFoundError(f"OpenXR environment GLB not found: {glb_path}")
        return glb_path, profile_path

    try:
        return resolve(selected_name)
    except (FileNotFoundError, ValueError) as exc:
        if selected_name.lower() == "default":
            raise
        print(
            f"[OpenXRViewer] Environment '{selected_name}' unavailable: {exc}; "
            "falling back to Default",
            flush=True,
        )
        return resolve("Default")


def _openxr_filament_config(settings: dict) -> dict[str, object]:
    """Resolve the selected packaged Filament scene for direct OpenXR runs."""
    src_root = Path(__file__).resolve().parents[1]
    platform_bridge = {
        "Windows": src_root / "xr_viewer" / "native" / "windows"
        / "filament_bridge.dll",
        "Linux": src_root / "xr_viewer" / "native" / "linux"
        / "libfilament_bridge.so",
        "Darwin": src_root / "xr_viewer" / "native" / "macos"
        / "libfilament_bridge.dylib",
    }.get(platform.system())
    glb_path, profile_path = _resolve_filament_environment_paths(
        settings,
        src_root,
    )

    bridge_path = os.environ.get("D2S_FILAMENT_BRIDGE") or (
        str(platform_bridge) if platform_bridge and platform_bridge.is_file() else None
    )
    configured_glb = os.environ.get("D2S_FILAMENT_GLB")
    configured_profile = os.environ.get("D2S_FILAMENT_PROFILE")
    return {
        "swapchain_color_mode": str(
            settings.get("OpenXR Color Mode", "sRGB")
        ).strip().lower(),
        "controller_model": str(settings.get("Controller Model", "PICO")),
        "filament_bridge_path": bridge_path,
        "filament_glb_path": configured_glb or (
            str(glb_path) if glb_path is not None else None
        ),
        "filament_profile_path": configured_profile
        or (str(profile_path) if profile_path is not None else None),
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


def _wait_for_runtime_ready(
    ready_event: threading.Event,
    pipeline_thread: threading.Thread,
) -> bool:
    print(
        "[Main] Waiting for inference load, first frame, and stereo warmup "
        "before OpenXR initialization...",
        flush=True,
    )
    while not shutdown_event.is_set():
        if ready_event.wait(0.05):
            print(
                "[Main] Inference pipeline ready; starting OpenXR "
                "Vulkan/Filament initialization",
                flush=True,
            )
            return True
        if not pipeline_thread.is_alive():
            raise RuntimeError(
                "Stereo pipeline stopped before inference startup completed"
            )
    return False


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
    runtime_ready_event = threading.Event()

    if str(RUN_MODE).strip().lower() == "openxr":
        # Keep source inference alive during the headset wake-up grace period.
        # The presenter enters hard idle after the configured 60-second timeout.
        context.openxr_state.bootstrap_done.set()
        context.openxr_state.source_active.set()

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
        runtime_ready_event=runtime_ready_event,
    )

    capture_thread = threading.Thread(
        target=CaptureSessionLoop(context.capture_config, capture_callbacks).run,
        args=(shutdown_event,),
        name="VulkanCapture",
        daemon=True,
    )
    pipeline = RuntimePipelineLoop(pipeline_context)
    print("[Main] Loading inference runtime before capture and OpenXR...", flush=True)
    pipeline.prepare()
    print("[Main] Inference runtime loaded", flush=True)
    pipeline_thread = threading.Thread(
        target=pipeline.run,
        name="VulkanStereoPipeline",
        daemon=True,
    )
    presenter = None
    presenter_thread = None
    output_consumer = None
    output_thread = None
    capture_thread.start()
    pipeline_thread.start()
    try:
        if str(RUN_MODE).strip().lower() == "openxr":
            if not _wait_for_runtime_ready(runtime_ready_event, pipeline_thread):
                return 0
            from xr_viewer.core_openxr_vulkan import (
                OpenXrVulkanConfig,
                OpenXrVulkanPresenter,
            )

            filament_config = _openxr_filament_config(settings)
            presenter = OpenXrVulkanPresenter(
                OpenXrVulkanConfig(**filament_config),
                on_headset_state=callbacks.on_openxr_headset_state,
                on_controller_shortcut=callbacks.on_openxr_controller_shortcut,
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
        print(
            f"Desktop2Stereo Vulkan runtime started: mode={RUN_MODE} device={DEVICE_INFO}",
            flush=True,
        )
        deadline = (
            None
            if max_seconds is None
            else time.monotonic() + max(0.0, max_seconds)
        )
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
