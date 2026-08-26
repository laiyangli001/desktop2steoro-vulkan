"""Assemble the Python capture and stereo pipeline for the Vulkan project."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import replace
from pathlib import Path

from capture import capture_frame_to_rgb, prepare_rgb_for_stereo_runtime
from capture.adaptive_rate import AdaptiveCaptureRate, adaptive_capture_enabled_for_mode
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
    LOCAL_VSYNC,
    MONITOR_INDEX,
    OPENXR_SCREEN_DISTANCE,
    OPENXR_SCREEN_WIDTH,
    OS_NAME,
    OUTPUT_RESOLUTION,
    RENDER_SIZE_CONFIG,
    RUN_MODE,
    SHOW_FPS,
    STEREO_DISPLAY_INDEX,
    STEREO_DISPLAY_SELECTION,
    WINDOW_TITLE,
    _get_settings,
    shutdown_event,
)
from utils.run_mode import normalize_run_mode
from utils.xr_headset_presets import DEFAULT_XR_HEADSET_MODEL
from streaming.stream_session import (
    CALIBRATABLE_STREAM_MODES,
    NetworkStreamSessionConfig,
    is_network_stream_mode,
    resolve_network_video_backend,
    supports_network_calibration,
)

from .runtime_callbacks import RuntimeCallbacks
from .runtime_context import (
    build_capture_callbacks,
    build_runtime_pipeline_context,
    create_runtime_context,
)
from .runtime_output import VulkanRuntimeOutputConsumer


def _resolve_filament_environment_paths(
    settings: dict,
    src_root: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    environment_name = str(settings.get("Environment Model", "")).strip()
    selected_name = (
        "Default"
        if not environment_name or environment_name.lower() == "none"
        else environment_name
    )
    environments_root = src_root / "xr_viewer" / "environments"

    def resolve(name: str) -> tuple[Path | None, Path, Path | None]:
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
        background = profile.get("background")
        background = background if isinstance(background, dict) else {}
        image_value = (
            background.get("image")
            or background.get("path")
            or background.get("file")
            or profile.get("background_image")
        )
        panorama_path = None
        if image_value:
            candidate = room_dir / str(image_value)
            if not candidate.is_file():
                raise FileNotFoundError(f"OpenXR environment panorama not found: {candidate}")
            panorama_path = candidate
        if glb_value in (None, "", False):
            return None, profile_path, panorama_path
        glb_path = room_dir / str(glb_value)
        if not glb_path.is_file():
            raise FileNotFoundError(f"OpenXR environment GLB not found: {glb_path}")
        return glb_path, profile_path, panorama_path

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


def _load_common_filament_defaults(src_root: Path) -> dict[str, object]:
    """Load shared environment defaults without making startup depend on them."""
    common_path = src_root / "xr_viewer" / "environments" / "common.json"
    try:
        common = json.loads(common_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(common, dict):
        return {}
    filament = common.get("filament", {})
    return filament if isinstance(filament, dict) else {}


def _resolve_openxr_render_scale(
    settings: dict,
    processing_size: int | tuple[int, int] | None = None,
) -> float:
    """Resolve the explicit OpenXR projection override.

    The GUI ``Render Scale`` belongs to capture/inference preprocessing.  It
    must not resize the OpenXR projection target, otherwise a 4K capture
    downscaled to 1K would take a different presentation path from native 1K.
    ``processing_size`` remains accepted for compatibility with callers and
    tests, but is intentionally not used for projection sizing.
    """
    env_value = os.environ.get("D2S_OPENXR_RENDER_SCALE")
    if env_value:
        try:
            return max(0.5, min(2.0, float(env_value)))
        except ValueError:
            pass
    try:
        return max(0.5, min(2.0, float(settings.get("OpenXR Render Scale", 1.0))))
    except (TypeError, ValueError):
        return 1.0


def _openxr_projection_config(settings: dict) -> dict[str, object]:
    """Resolve OpenXR presentation settings independently of Filament."""
    return {
        "render_scale": _resolve_openxr_render_scale(settings),
        "swapchain_color_mode": str(
            settings.get("OpenXR Color Mode", "sRGB")
        ).strip().lower(),
        "controller_model": str(settings.get("Controller Model", "PICO")),
        "headset_model": str(
            settings.get("XR Headset Model", DEFAULT_XR_HEADSET_MODEL)
        ),
        "monitor_index": max(1, int(settings.get("Monitor Index", 1) or 1)),
    }


def _openxr_filament_config(
    settings: dict,
) -> dict[str, object]:
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
    glb_path, profile_path, panorama_path = _resolve_filament_environment_paths(
        settings,
        src_root,
    )
    common_filament = _load_common_filament_defaults(src_root)
    environment_name = str(settings.get("Environment Model", "Default")).strip()
    default_environment = not environment_name or environment_name.lower() in {"default", "none"}

    bridge_path = os.environ.get("D2S_FILAMENT_BRIDGE") or (
        str(platform_bridge) if platform_bridge and platform_bridge.is_file() else None
    )
    configured_glb = os.environ.get("D2S_FILAMENT_GLB")
    configured_profile = os.environ.get("D2S_FILAMENT_PROFILE")
    return {
        "filament_bridge_path": bridge_path,
        "filament_glb_path": configured_glb or (
            str(glb_path) if glb_path is not None else None
        ),
        "filament_profile_path": configured_profile
        or (str(profile_path) if profile_path is not None else None),
        "filament_panorama_path": os.environ.get("D2S_FILAMENT_PANORAMA")
        or (str(panorama_path) if panorama_path is not None else None),
        "filament_scene_exposure_ev": float(
            settings.get(
                "Filament Scene Exposure",
                common_filament.get("scene_exposure_ev", 2.0),
            )
        ),
        "filament_skybox_brightness": float(
            settings.get(
                "Filament Skybox Brightness",
                common_filament.get("skybox_brightness", 1.0),
            )
        ),
        "filament_ambient_light_color": tuple(
            common_filament.get("ambient_light_color", (0.14, 0.13, 0.15))
        ),
        "filament_ambient_light_intensity_lux": float(
            common_filament.get("ambient_light_intensity_lux", 30000.0)
        ),
        "filament_controller_ambient_light_intensity_lux": float(
            common_filament.get("controller_ambient_light_intensity_lux", 8000.0)
        ),
        "filament_controller_hdr_ambient_light_intensity_lux": float(
            common_filament.get("controller_hdr_ambient_light_intensity_lux", 8000.0)
        ),
        "filament_controller_light_intensity_candela": float(
            common_filament.get("controller_light_intensity_candela", 2000.0)
        ),
        "filament_fill_light_color": tuple(
            common_filament.get("controller_head_light_color", (0.55, 0.55, 0.58))
        ),
        "filament_controller_head_light_weight": float(
            common_filament.get("controller_head_light_weight", 0.85)
        ),
        "filament_controller_top_light_weight": float(
            common_filament.get("controller_top_light_weight", 0.6)
        ),
        "filament_controller_top_light_color": tuple(
            common_filament.get("controller_top_light_color", (0.95, 0.97, 1.0))
        ),
        "filament_controller_head_light_offset": tuple(
            common_filament.get("controller_head_light_offset", (0.0, 0.05, 0.0))
        ),
        "filament_controller_top_light_offset": tuple(
            common_filament.get("controller_top_light_offset", (0.0, 0.45, -0.18))
        ),
        "filament_controller_head_light_falloff": float(
            common_filament.get("controller_head_light_falloff", 2.0)
        ),
        "filament_controller_top_light_falloff": float(
            common_filament.get("controller_top_light_falloff", 2.0)
        ),
        "filament_controller_head_light_cast_shadows": bool(
            common_filament.get("controller_head_light_cast_shadows", False)
        ),
        "filament_controller_top_light_cast_shadows": bool(
            common_filament.get("controller_top_light_cast_shadows", False)
        ),
        "filament_controller_screen_light_enabled": bool(
            common_filament.get("controller_screen_light_enabled", True)
        ),
        "filament_controller_screen_light_intensity_lux": float(
            common_filament.get("controller_screen_light_intensity_lux", 500.0)
        ),
        "filament_controller_screen_light_saturation": float(
            common_filament.get("controller_screen_light_saturation", 0.65)
        ),
        "filament_controller_screen_light_max_luminance": float(
            common_filament.get("controller_screen_light_max_luminance", 0.40)
        ),
        "filament_controller_screen_light_smoothing_seconds": float(
            common_filament.get("controller_screen_light_smoothing_seconds", 0.18)
        ),
        "filament_controller_screen_light_sample_hz": float(
            common_filament.get("controller_screen_light_sample_hz", 12.0)
        ),
        "filament_controller_screen_light_cast_shadows": bool(
            common_filament.get("controller_screen_light_cast_shadows", False)
        ),
        "filament_environment_screen_light_enabled": bool(
            False
            if default_environment
            else common_filament.get("environment_screen_light_enabled", True)
        ),
        "filament_environment_screen_light_intensity_candela": float(
            common_filament.get("environment_screen_light_intensity_candela", 120.0)
        ),
        "filament_environment_screen_light_saturation": float(
            common_filament.get("environment_screen_light_saturation", 0.70)
        ),
        "filament_environment_screen_light_max_luminance": float(
            common_filament.get("environment_screen_light_max_luminance", 0.40)
        ),
        "filament_environment_screen_light_smoothing_seconds": float(
            common_filament.get("environment_screen_light_smoothing_seconds", 0.18)
        ),
        "filament_environment_screen_light_sample_hz": float(
            common_filament.get("environment_screen_light_sample_hz", 12.0)
        ),
        "filament_environment_screen_light_falloff": float(
            common_filament.get("environment_screen_light_falloff", 4.0)
        ),
        "filament_environment_screen_light_offset": float(
            common_filament.get("environment_screen_light_offset", 0.08)
        ),
        "filament_environment_screen_light_cast_shadows": bool(
            common_filament.get("environment_screen_light_cast_shadows", False)
        ),
        "filament_glow_sample_hz": float(
            common_filament.get("glow_sample_hz", 30.0)
        ),
        "filament_glow_smoothing_seconds": float(
            common_filament.get("glow_smoothing_seconds", 0.10)
        ),
        # These values are resolved by the legacy viewer-settings path and
        # exported through utils; keep the Vulkan entrypoint as a consumer.
        "filament_screen_width": float(OPENXR_SCREEN_WIDTH),
        "filament_screen_distance": float(OPENXR_SCREEN_DISTANCE),
    }


def _exclude_local_output_from_capture(settings: dict, *, os_name: str) -> bool:
    return (
        str(os_name).strip().lower() == "windows"
        and settings.get("Run Mode", "Local Viewer") == "3D Monitor"
    )


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
    configured_run_mode = normalize_run_mode(
        settings.get("Run Mode", "Local Viewer")
    )
    direct_stream_mode = is_network_stream_mode(configured_run_mode) or configured_run_mode == "MJPEG Streamer"
    if direct_stream_mode:
        os.environ["D2S_RUNTIME_OUTPUT_UINT8"] = "1"
    try:
        configured_target_fps = int(settings.get("Target FPS", 0) or 0)
    except (TypeError, ValueError):
        configured_target_fps = 0
    adaptive_capture_rate = AdaptiveCaptureRate(
        FPS,
        enabled=adaptive_capture_enabled_for_mode(
            configured_run_mode, configured_target_fps
        ),
    )
    if is_network_stream_mode(configured_run_mode):
        probe_capture_fps = adaptive_capture_rate.begin_stream_probe(int(FPS))
        print(
            "[DirectSbsStream] Stream-rate probe capture headroom: "
            f"requested={int(FPS)} capture={probe_capture_fps} FPS",
            flush=True,
        )
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
        capture_fps_provider=adaptive_capture_rate.current_fps,
    )
    callbacks = RuntimeCallbacks(context, show_fps=bool(SHOW_FPS))

    def observe_sbs_fps(sbs_fps, frame_count=None):
        return adaptive_capture_rate.observe_sbs_fps(
            sbs_fps,
            capture_fps=callbacks.capture_fps(),
            frame_count=frame_count,
        )

    callbacks.breakdown_set_latest(
        "adaptive_capture_target_fps", adaptive_capture_rate.current_fps()
    )
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

    presenter = None
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
        openxr_presenter_pressure=lambda: bool(
            presenter is not None and presenter.inference_backpressure_active()
        ),
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
    presenter_thread = None
    output_consumer = None
    output_thread = None
    local_viewer_thread = None
    network_output = None
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

            presenter_config = _openxr_projection_config(settings)
            presenter_config.update(_openxr_filament_config(settings))
            presenter = OpenXrVulkanPresenter(
                OpenXrVulkanConfig(**presenter_config),
                on_headset_state=callbacks.on_openxr_headset_state,
                on_controller_shortcut=callbacks.on_openxr_controller_shortcut,
                on_breakdown_inc=callbacks.breakdown_inc,
                on_breakdown_add_time=callbacks.breakdown_add_time,
                on_breakdown_set_latest=callbacks.breakdown_set_latest,
                on_runtime_fps=callbacks.runtime_fps,
                on_capture_fps=callbacks.capture_fps,
                on_sbs_fps=observe_sbs_fps if adaptive_capture_rate.enabled else None,
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
            )
            output_thread = threading.Thread(
                target=output_consumer.run,
                name="VulkanOutputConsumer",
                daemon=True,
            )
            output_thread.start()
        elif configured_run_mode in {
            "RTMP Streamer",
            "MJPEG Streamer",
        }:
            from streaming.direct_sbs import (
                DirectSbsOutputConsumer,
                AmdAmfDirectSbsOutput,
                FfmpegDirectSbsOutput,
                IntelD3D11DirectSbsOutput,
                IntelQsvDirectSbsOutput,
                MjpegDirectSbsOutput,
                PyNvDirectSbsOutput,
                VulkanDirectSbsOutput,
            )
            from streaming.stream_calibration import build_calibration_fingerprint

            if configured_run_mode in CALIBRATABLE_STREAM_MODES:
                audio_backend = str(
                    settings.get("Audio Capture Backend", "auto") or "auto"
                ).strip().casefold()
                selected_audio = str(settings.get("Stereo Mix", "") or "").strip()
                if audio_backend in {"auto", "soundcard"} and not selected_audio.casefold().startswith(
                    ("soundcard:", "wasapi:")
                ):
                    selected_audio = f"soundcard:{selected_audio}"
                stream_config = replace(
                    NetworkStreamSessionConfig.from_settings(settings, fps=int(FPS)),
                    stereo_mix_device=selected_audio,
                )
                output_kwargs = dict(
                    base_dir=context.base_dir,
                    protocol=stream_config.protocol,
                    port=stream_config.port,
                    stream_key=stream_config.stream_key,
                    fps=stream_config.fps,
                    crf=stream_config.crf,
                    stereo_mix_device=stream_config.stereo_mix_device,
                    audio_delay=stream_config.audio_delay,
                    os_name=OS_NAME,
                    prefer_nvenc=(
                        OS_NAME in {"Windows", "Linux"}
                        and "NVIDIA" in str(DEVICE_INFO).upper()
                    ),
                    display_mode=stream_config.display_mode,
                    target_bitrate_mbps=(
                        stream_config.target_bitrate_mbps
                        if bool(settings.get("Use Stream Calibration", True))
                        else 0
                    ),
                    peak_bitrate_mbps=(
                        stream_config.peak_bitrate_mbps
                        if bool(settings.get("Use Stream Calibration", True))
                        else 0
                    ),
                    auto_calibration=(
                        supports_network_calibration(
                            configured_run_mode,
                            settings.get("Stream Protocol", "WebRTC"),
                        )
                        and os.environ.get("D2S_STREAM_CALIBRATE", "0") == "1"
                    ),
                    calibration_port=int(
                        settings.get(
                            "Stream Calibration Port",
                            min(65535, stream_config.port + 1),
                        )
                    ),
                    on_calibration_fps=adaptive_capture_rate.set_calibration_limit,
                    calibration_fingerprint=build_calibration_fingerprint(settings),
                    on_stream_fps_selected=adaptive_capture_rate.finish_stream_probe,
                )
                backend_decision = resolve_network_video_backend(
                    configured_run_mode,
                    settings.get("Video Encoder Backend", "auto"),
                    device_info=DEVICE_INFO,
                )
                video_backend = backend_decision.backend
                has_nvidia_gpu = "NVIDIA" in str(DEVICE_INFO).upper()
                has_amd_gpu = any(
                    token in str(DEVICE_INFO).upper()
                    for token in ("AMD", "RADEON")
                )
                has_intel_gpu = "INTEL" in str(DEVICE_INFO).upper()
                print(
                    f"[DirectSbsStream] mode={configured_run_mode} "
                    f"encoder={video_backend} ({backend_decision.reason})",
                    flush=True,
                )
                network_output = FfmpegDirectSbsOutput(**output_kwargs)
                if video_backend == "auto":
                    network_output.close()
                    if has_intel_gpu:
                        # Intel owns its D3D11/oneVPL vendor path first; its
                        # native sink already reuses the Vulkan packed-SBS
                        # bridge when the shared-surface contract is available.
                        network_output = IntelD3D11DirectSbsOutput(**output_kwargs)
                    else:
                        # NVIDIA/AMD Auto enters the same lazy output at the
                        # vendor-native stage, then proceeds to Vulkan and the
                        # portable OpenGL/FFmpeg fallbacks.
                        network_output = VulkanDirectSbsOutput(
                            **output_kwargs, vendor_gpu_first=True
                        )
                elif video_backend in {"intel", "qsv"}:
                    network_output.close()
                    network_output = (
                        IntelD3D11DirectSbsOutput(**output_kwargs)
                        if video_backend == "intel"
                        else IntelQsvDirectSbsOutput(**output_kwargs)
                    )
                elif video_backend == "vulkan":
                    network_output.close()
                    network_output = VulkanDirectSbsOutput(**output_kwargs)
                elif video_backend == "pynv" and has_nvidia_gpu:
                    try:
                        network_output.close()
                        network_output = PyNvDirectSbsOutput(**output_kwargs)
                    except Exception as exc:
                        print(
                            f"[DirectSbsStream] PyNvVideoCodec startup unavailable: {exc}; "
                            "falling back to FFmpeg without changing MediaMTX settings",
                            flush=True,
                        )
                        network_output = FfmpegDirectSbsOutput(**output_kwargs)
                elif video_backend == "amd" and has_amd_gpu and not selected_audio:
                    try:
                        network_output.close()
                        network_output = AmdAmfDirectSbsOutput(**output_kwargs)
                    except Exception as exc:
                        print(
                            f"[DirectSbsStream] AMD AMF bridge unavailable: {exc}; "
                            "falling back to FFmpeg without changing MediaMTX settings",
                            flush=True,
                        )
                        network_output = FfmpegDirectSbsOutput(**output_kwargs)
                elif video_backend == "amd" and not has_amd_gpu:
                    print(
                        "[DirectSbsStream] AMD AMF backend requires an AMD/Radeon GPU; "
                        "falling back to FFmpeg hardware/software encoder",
                        flush=True,
                    )
                elif video_backend == "amd" and selected_audio:
                    print(
                        "[DirectSbsStream] native AMD AMF path requires audio disabled; "
                        "falling back to FFmpeg to preserve audio",
                        flush=True,
                    )
                elif video_backend == "pynv" and not has_nvidia_gpu:
                    print(
                        "[DirectSbsStream] PyNvVideoCodec is NVIDIA-only; "
                        "falling back to FFmpeg hardware/software encoder",
                        flush=True,
                    )
                if (
                    isinstance(network_output, IntelD3D11DirectSbsOutput)
                    and str(os.environ.get("D2S_ONEVPL_FINAL_SBS", "0")).strip().casefold()
                    in {"1", "true", "yes", "on"}
                ):
                    # StereoRuntime defers the supported Vulkan request to the
                    # Intel sink, which owns the Vulkan image ring and the
                    # D3D11/oneVPL lifetime.
                    os.environ.setdefault("D2S_INTEL_VULKAN_SBS", "1")
            else:
                network_output = MjpegDirectSbsOutput(
                    port=int(settings.get("Streamer Port", 1122)),
                    fps=int(FPS),
                    quality=int(settings.get("Stream Quality", 90)),
                )
            callbacks.set_stream_output(network_output)
            network_output.start()
            output_consumer = DirectSbsOutputConsumer(
                runtime_q=context.runtime_q,
                shutdown_event=shutdown_event,
                output=network_output,
                source_stat_inc=callbacks.source_stat_inc,
                show_fps_provider=callbacks.show_fps,
                on_sbs_fps=(
                    observe_sbs_fps if adaptive_capture_rate.enabled else None
                ),
                fps_report_interval=(
                    1.0
                    if str(getattr(network_output, "protocol", "")).strip().upper()
                    != "MJPEG"
                    else 5.0
                ),
            )
            output_thread = threading.Thread(
                target=output_consumer.run,
                name="DirectSbsOutputConsumer",
                daemon=True,
            )
            output_thread.start()
        elif str(RUN_MODE).strip().lower() == "viewer":
            # Local Viewer no longer falls through as an unconsumed runtime_q.
            # It owns a GLFW Vulkan surface and presents the already packed SBS.
            from viewer.vulkan_local_viewer import (
                VulkanLocalViewerConfig,
                run_vulkan_local_viewer,
            )

            selected_monitor = (
                int(STEREO_DISPLAY_INDEX)
                if bool(STEREO_DISPLAY_SELECTION)
                else int(MONITOR_INDEX)
            )
            local_viewer_config = VulkanLocalViewerConfig(
                title=f"{WINDOW_TITLE or 'Desktop2Stereo'} Vulkan Viewer",
                monitor_index=max(0, selected_monitor),
                fullscreen=bool(STEREO_DISPLAY_SELECTION),
                window_preview=bool(settings.get("Window Preview", False)),
                preview_monitor_index=max(0, int(MONITOR_INDEX)),
                exclude_from_capture=_exclude_local_output_from_capture(
                    settings,
                    os_name=OS_NAME,
                ),
                vsync=bool(LOCAL_VSYNC),
                show_fps=bool(SHOW_FPS),
                show_fps_provider=callbacks.show_fps,
                on_sbs_fps=observe_sbs_fps if adaptive_capture_rate.enabled else None,
                on_breakdown_inc=callbacks.breakdown_inc,
                on_breakdown_add_time=callbacks.breakdown_add_time,
            )
            local_viewer_thread = threading.Thread(
                target=run_vulkan_local_viewer,
                kwargs={
                    "runtime_q": context.runtime_q,
                    "shutdown_event": shutdown_event,
                    "config": local_viewer_config,
                },
                name="VulkanLocalViewer",
                daemon=True,
            )
            local_viewer_thread.start()
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
            close_output_consumer = getattr(output_consumer, "close", None)
            if callable(close_output_consumer):
                close_output_consumer()
        if network_output is not None:
            network_output.close()
        if presenter_thread is not None:
            # run_until owns Filament/Vulkan teardown on the Presenter thread.
            # Do not let the main thread race that teardown after a timeout.
            presenter_thread.join()
        if local_viewer_thread is not None:
            local_viewer_thread.join(timeout=2.0)
        if presenter is not None:
            presenter.close()
        close = getattr(context.stereo_runtime, "close", None)
        if callable(close):
            close()
    return 0
