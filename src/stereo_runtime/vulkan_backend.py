"""Runtime adapter for the fused Vulkan stereo compute pass."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from viewer.vulkan_context import ImageState, VulkanContext, VulkanContextConfig
from viewer.vulkan_descriptors import VulkanStorageBuffer
from viewer.vulkan_resources import VulkanExportableBuffer, VulkanExportableSemaphore

from .vulkan_stereo_pass import (
    VulkanLayeredStereoParams,
    VulkanLayeredStereoPass,
    VulkanStereoFusedParams,
    VulkanStereoFusedPass,
)
from .vulkan_stereo_image_pass import VulkanStereoImagePass


class VulkanStereoBackendUnavailable(RuntimeError):
    """Raised when the runtime Vulkan fused pass cannot be created."""


class VulkanStereoImageComputeBackend:
    """Run stereo Compute directly into presenter-owned Vulkan images.

    CUDA input tensors are copied directly into imported Vulkan storage buffers
    when CUDA external-memory interop is available. The host-visible path is
    retained as a fallback for unsupported devices.
    """

    def __init__(self, context: Any, *, shader_path: str | Path | None = None) -> None:
        self.context = context
        self.shader_path = Path(shader_path) if shader_path is not None else (
            Path(__file__).resolve().parents[2]
            / "shaders"
            / "d2s_stereo_layered_output.spv"
        )
        self._pass: VulkanStereoImagePass | None = None
        self._buffers: tuple[VulkanStorageBuffer, ...] = ()
        self._shape: tuple[int, int] | None = None
        self._last_submit_timeline = 0
        self._frame_id = 0
        self._closed = False
        self._cuda_importer = None
        self._cuda_input_buffers: tuple[VulkanExportableBuffer, ...] = ()
        self._cuda_input_ready: VulkanExportableSemaphore | None = None
        self._cuda_input_error: str | None = None

    @property
    def device_name(self) -> str:
        return str(getattr(getattr(self.context, "device_info", None), "name", "unknown"))

    @staticmethod
    def _validate_inputs(rgb: torch.Tensor, depth: torch.Tensor) -> tuple[int, int]:
        if not isinstance(rgb, torch.Tensor) or not isinstance(depth, torch.Tensor):
            raise TypeError("Vulkan stereo image backend requires torch tensors")
        if rgb.ndim != 4 or int(rgb.shape[0]) != 1 or int(rgb.shape[1]) != 3:
            raise ValueError(f"expected RGB [1,3,H,W], got {tuple(rgb.shape)}")
        if depth.ndim != 4 or int(depth.shape[0]) != 1 or int(depth.shape[1]) != 1:
            raise ValueError(f"expected depth [1,1,H,W], got {tuple(depth.shape)}")
        if tuple(rgb.shape[-2:]) != tuple(depth.shape[-2:]):
            raise ValueError("Vulkan stereo RGB and depth dimensions must match")
        return int(rgb.shape[-2]), int(rgb.shape[-1])

    def _ensure_shape(self, height: int, width: int) -> None:
        shape = (int(height), int(width))
        if self._shape == shape and self._pass is not None:
            return
        self._close_resources()
        if self._closed or self.context is None or getattr(self.context, "closed", False):
            raise VulkanStereoBackendUnavailable("presenter Vulkan context is closed")
        self._pass = VulkanStereoImagePass(
            self.context,
            width=shape[1],
            height=shape[0],
            shader_path=self.shader_path,
        )
        sizes = self._pass.input_buffer_sizes
        self._buffers = tuple(
            VulkanStorageBuffer(self.context, sizes[name]) for name in ("rgb", "depth")
        )
        self._shape = shape

    def submit_to_images(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        left_eye: Any,
        right_eye: Any,
        *,
        params: VulkanLayeredStereoParams,
        ready_timeline: int | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if self._closed:
            raise VulkanStereoBackendUnavailable("Vulkan stereo image backend is closed")
        height, width = self._validate_inputs(rgb, depth)
        self._ensure_shape(height, width)
        if self._pass is None or len(self._buffers) != 2:
            raise VulkanStereoBackendUnavailable("Vulkan stereo image pass is unavailable")
        for image in (left_eye, right_eye):
            if getattr(image, "context", None) is not self.context:
                raise ValueError("stereo output image belongs to a different Vulkan context")
            if getattr(image, "resource", None) is not None:
                image = image.resource
            if int(getattr(image, "width", 0)) != width or int(getattr(image, "height", 0)) != height:
                raise ValueError("stereo output image dimensions do not match")

        # Input buffers are reused only after the preceding dispatch has
        # completed. This wait does not wait for Filament; it protects the
        # persistent input storage from a data race.
        if self._last_submit_timeline:
            self.context.wait_for_timeline(self._last_submit_timeline)

        input_mode = "host_visible_buffer"
        wait_semaphore = None
        input_upload_start = time.perf_counter()
        use_cuda_input = (
            str(getattr(getattr(rgb, "device", None), "type", "")) == "cuda"
            and str(getattr(rgb, "dtype", "")) == "torch.float32"
            and bool(getattr(rgb, "is_contiguous", lambda: False)())
            and str(getattr(getattr(depth, "device", None), "type", "")) == "cuda"
            and str(getattr(depth, "dtype", "")) == "torch.float32"
            and bool(getattr(depth, "is_contiguous", lambda: False)())
        )
        if use_cuda_input:
            try:
                self._ensure_cuda_inputs(height, width)
                importer = self._cuda_importer
                ready = self._cuda_input_ready
                if importer is None or ready is None:
                    raise VulkanStereoBackendUnavailable(
                        "CUDA Vulkan input interop is unavailable"
                    )
                stream = int(torch.cuda.current_stream(device=rgb.device).cuda_stream)
                importer.copy_tensor_to_buffer(
                    rgb, self._cuda_input_buffers[0], stream=stream
                )
                importer.copy_tensor_to_buffer(
                    depth, self._cuda_input_buffers[1], stream=stream
                )
                importer.signal_semaphore(ready, stream=stream)
                wait_semaphore = ready.semaphore
                input_mode = "cuda_external_buffer"
                self._cuda_input_error = None
            except Exception as exc:
                self._cuda_input_error = f"{type(exc).__name__}: {exc}"
                self._close_cuda_inputs()

        if input_mode == "cuda_external_buffer":
            input_upload_ms = (time.perf_counter() - input_upload_start) * 1000.0
            input_buffers = self._cuda_input_buffers
        else:
            input_upload_start = time.perf_counter()
            self._buffers[0].write_bytes(
                VulkanStereoComputeBackend._planar_bytes(rgb, channels=3)
            )
            self._buffers[1].write_bytes(
                VulkanStereoComputeBackend._planar_bytes(depth, channels=1)
            )
            input_upload_ms = (time.perf_counter() - input_upload_start) * 1000.0
            input_buffers = self._buffers
        timeline = self._pass.submit(
            input_buffers[0],
            input_buffers[1],
            left_eye,
            right_eye,
            params=params,
            frame_id=self._frame_id,
            config_version=0,
            ready_timeline=ready_timeline,
            wait_semaphore=wait_semaphore,
        )
        self._frame_id += 1
        self._last_submit_timeline = int(timeline)
        vk = self.context.vk
        output_state = ImageState(
            layout=vk.VK_IMAGE_LAYOUT_GENERAL,
            access_mask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            stage_mask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            queue_family_index=self.context.queue_family_index,
        )
        for image in (left_eye, right_eye):
            resource = getattr(image, "resource", image)
            self.context.register_image_state(resource.image, output_state)
        return int(timeline), {
            "stereo_compute_backend": "vulkan",
            "vulkan_fused_backend": "vulkan_stereo_layered_output_image",
            "vulkan_device": self.device_name,
            "vulkan_submit_timeline": int(timeline),
            "vulkan_readback": "none",
            "vulkan_compute_output": "presenter_owned_storage_image",
            "vulkan_compute_input_color_space": "srgb",
            "vulkan_compute_output_image_format": "R8G8B8A8_UNORM",
            "vulkan_compute_output_image_encoding": "linear",
            "vulkan_output_sync": "vulkan_compute_external_semaphore",
            "vulkan_input_path": input_mode,
            "vulkan_input_upload_ms": input_upload_ms,
            "vulkan_input_error": self._cuda_input_error,
        }

    def _ensure_cuda_inputs(self, height: int, width: int) -> None:
        if (
            self._cuda_importer is not None
            and self._cuda_input_ready is not None
            and len(self._cuda_input_buffers) == 2
            and self._shape == (int(height), int(width))
        ):
            return
        self._close_cuda_inputs()
        from viewer.cuda_vulkan_interop import CudaVulkanImageImporter

        if self._pass is None:
            raise VulkanStereoBackendUnavailable("Vulkan stereo image pass is unavailable")
        importer = CudaVulkanImageImporter()
        sizes = self._pass.input_buffer_sizes
        buffers = tuple(
            VulkanExportableBuffer(
                self.context,
                int(sizes[name]),
                label=f"stereo-input-{name}",
            )
            for name in ("rgb", "depth")
        )
        ready = VulkanExportableSemaphore(self.context, label="stereo-input-ready")
        try:
            for buffer in buffers:
                importer.register_buffer(buffer)
            importer.register_semaphore(ready)
        except Exception:
            ready.close()
            for buffer in buffers:
                buffer.close()
            importer.close()
            raise
        self._cuda_importer = importer
        self._cuda_input_buffers = buffers
        self._cuda_input_ready = ready

    def _close_cuda_inputs(self) -> None:
        if self._cuda_importer is not None:
            self._cuda_importer.close()
        self._cuda_importer = None
        if self._cuda_input_ready is not None:
            self._cuda_input_ready.close()
        self._cuda_input_ready = None
        for buffer in self._cuda_input_buffers:
            buffer.close()
        self._cuda_input_buffers = ()

    def _close_resources(self) -> None:
        self._close_cuda_inputs()
        if self.context is not None and not getattr(self.context, "closed", False):
            if self._last_submit_timeline:
                self.context.wait_for_timeline(self._last_submit_timeline)
        for buffer in self._buffers:
            buffer.close()
        self._buffers = ()
        if self._pass is not None:
            self._pass.close()
        self._pass = None
        self._shape = None
        self._last_submit_timeline = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_resources()

    def __enter__(self) -> "VulkanStereoImageComputeBackend":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class VulkanStereoComputeBackend:
    """Execute one fused stereo frame on Vulkan and return torch tensors.

    The first production integration deliberately uses host-visible storage
    buffers at the runtime boundary. This keeps the pass usable from the
    inference thread while the later presenter-owned zero-copy image path is
    developed independently.
    """

    def __init__(
        self,
        *,
        context: Any | None = None,
        shader_path: str | Path | None = None,
    ) -> None:
        self._owns_context = context is None
        self.context = context
        self.shader_path = Path(shader_path) if shader_path is not None else (
            Path(__file__).resolve().parents[2] / "shaders" / "d2s_stereo_fused.spv"
        )
        self._pass: VulkanStereoFusedPass | None = None
        self._buffers: tuple[VulkanStorageBuffer, ...] = ()
        self._shape: tuple[int, int] | None = None
        self._layered_pass: VulkanLayeredStereoPass | None = None
        self._layered_buffers: tuple[VulkanStorageBuffer, ...] = ()
        self._layered_shape: tuple[int, int] | None = None
        self._frame_id = 0
        self._closed = False

        try:
            if self.context is None:
                self.context = VulkanContext.create(
                    VulkanContextConfig(frame_context_count=3)
                )
        except Exception as exc:
            self.close()
            raise VulkanStereoBackendUnavailable(
                f"unable to create Vulkan Compute context: {type(exc).__name__}: {exc}"
            ) from exc

    @property
    def device_name(self) -> str:
        return str(getattr(getattr(self.context, "device_info", None), "name", "unknown"))

    def _ensure_shape(self, height: int, width: int) -> None:
        shape = (int(height), int(width))
        if self._shape == shape and self._pass is not None:
            return
        self._close_pass_resources()
        if self.context is None or getattr(self.context, "closed", False):
            raise VulkanStereoBackendUnavailable("Vulkan Compute context is closed")
        self._pass = VulkanStereoFusedPass(
            self.context,
            width=shape[1],
            height=shape[0],
            shader_path=self.shader_path,
        )
        sizes = self._pass.buffer_sizes
        self._buffers = tuple(
            VulkanStorageBuffer(self.context, sizes[name])
            for name in ("rgb", "depth", "left_eye", "right_eye", "occlusion_mask")
        )
        self._shape = shape

    def _ensure_layered_shape(self, height: int, width: int) -> None:
        shape = (int(height), int(width))
        if self._layered_shape == shape and self._layered_pass is not None:
            return
        self._close_pass_resources()
        if self.context is None or getattr(self.context, "closed", False):
            raise VulkanStereoBackendUnavailable("Vulkan Compute context is closed")
        shader_path = self.shader_path.with_name("d2s_stereo_layered.spv")
        self._layered_pass = VulkanLayeredStereoPass(
            self.context,
            width=shape[1],
            height=shape[0],
            shader_path=shader_path,
        )
        sizes = self._layered_pass.buffer_sizes
        self._layered_buffers = tuple(
            VulkanStorageBuffer(self.context, sizes[name])
            for name in ("rgb", "depth", "left_eye", "right_eye", "occlusion_mask")
        )
        self._layered_shape = shape

    @staticmethod
    def _planar_bytes(tensor: torch.Tensor, *, channels: int) -> bytes:
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if value.ndim == 4:
            if int(value.shape[0]) != 1 or int(value.shape[1]) != channels:
                raise ValueError(
                    f"expected a single {channels}-channel BCHW tensor, got {tuple(value.shape)}"
                )
            value = value[0]
        elif value.ndim == 3:
            if int(value.shape[0]) != channels:
                raise ValueError(
                    f"expected a {channels}-channel CHW tensor, got {tuple(value.shape)}"
                )
        else:
            raise ValueError(f"expected a BCHW or CHW tensor, got {tuple(value.shape)}")
        return value.numpy().tobytes(order="C")

    def submit_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        *,
        params: VulkanStereoFusedParams,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._closed:
            raise VulkanStereoBackendUnavailable("Vulkan stereo backend is closed")
        if not isinstance(rgb, torch.Tensor) or not isinstance(depth, torch.Tensor):
            raise TypeError("Vulkan stereo backend requires torch tensors")
        if rgb.ndim != 4 or int(rgb.shape[0]) != 1 or int(rgb.shape[1]) != 3:
            raise ValueError(f"Vulkan stereo backend requires RGB shape [1,3,H,W], got {tuple(rgb.shape)}")
        if depth.ndim != 4 or int(depth.shape[0]) != 1 or int(depth.shape[1]) != 1:
            raise ValueError(f"Vulkan stereo backend requires depth shape [1,1,H,W], got {tuple(depth.shape)}")
        if tuple(rgb.shape[-2:]) != tuple(depth.shape[-2:]):
            raise ValueError("Vulkan stereo RGB and depth dimensions must match")

        height, width = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        self._ensure_shape(height, width)
        total_start = time.perf_counter()
        upload_start = time.perf_counter()
        rgb_bytes = self._planar_bytes(rgb, channels=3)
        depth_bytes = self._planar_bytes(depth, channels=1)
        self._buffers[0].write_bytes(rgb_bytes)
        self._buffers[1].write_bytes(depth_bytes)
        upload_ms = (time.perf_counter() - upload_start) * 1000.0
        timeline = self._pass.submit(
            *self._buffers,
            params=params,
            frame_id=self._frame_id,
            config_version=0,
        )
        self._frame_id += 1
        # The host-visible readback is intentional for this first integration;
        # it provides a correct fallback before presenter-owned image interop.
        wait_start = time.perf_counter()
        self.context.wait_idle()
        submit_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        readback_start = time.perf_counter()
        pixel_count = width * height
        left = self._read_float_buffer(self._buffers[2], 3 * pixel_count).reshape(1, 3, height, width)
        right = self._read_float_buffer(self._buffers[3], 3 * pixel_count).reshape(1, 3, height, width)
        mask = self._read_float_buffer(self._buffers[4], pixel_count).reshape(1, 1, height, width)
        left = left.to(device=rgb.device)
        right = right.to(device=rgb.device)
        mask = mask.to(device=rgb.device)
        readback_ms = (time.perf_counter() - readback_start) * 1000.0
        debug = {
            "stereo_compute_backend": "vulkan",
            "vulkan_fused_backend": "vulkan_stereo_fused",
            "vulkan_device": self.device_name,
            "vulkan_submit_timeline": int(timeline),
            "vulkan_readback": "host_visible_storage_buffer",
            "vulkan_host_upload_ms": upload_ms,
            "vulkan_submit_wait_ms": submit_wait_ms,
            "vulkan_host_readback_ms": readback_ms,
            "vulkan_total_ms": (time.perf_counter() - total_start) * 1000.0,
        }
        return left, right, mask, debug

    def submit_layered_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        *,
        params: VulkanLayeredStereoParams,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._closed:
            raise VulkanStereoBackendUnavailable("Vulkan stereo backend is closed")
        if not isinstance(rgb, torch.Tensor) or not isinstance(depth, torch.Tensor):
            raise TypeError("Vulkan stereo backend requires torch tensors")
        if rgb.ndim != 4 or int(rgb.shape[0]) != 1 or int(rgb.shape[1]) != 3:
            raise ValueError(f"Vulkan stereo backend requires RGB shape [1,3,H,W], got {tuple(rgb.shape)}")
        if depth.ndim != 4 or int(depth.shape[0]) != 1 or int(depth.shape[1]) != 1:
            raise ValueError(f"Vulkan stereo backend requires depth shape [1,1,H,W], got {tuple(depth.shape)}")
        if tuple(rgb.shape[-2:]) != tuple(depth.shape[-2:]):
            raise ValueError("Vulkan stereo RGB and depth dimensions must match")

        height, width = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        self._ensure_layered_shape(height, width)
        total_start = time.perf_counter()
        upload_start = time.perf_counter()
        rgb_bytes = self._planar_bytes(rgb, channels=3)
        depth_bytes = self._planar_bytes(depth, channels=1)
        self._layered_buffers[0].write_bytes(rgb_bytes)
        self._layered_buffers[1].write_bytes(depth_bytes)
        upload_ms = (time.perf_counter() - upload_start) * 1000.0
        timeline = self._layered_pass.submit(
            *self._layered_buffers,
            params=params,
            frame_id=self._frame_id,
            config_version=0,
        )
        self._frame_id += 1
        wait_start = time.perf_counter()
        self.context.wait_idle()
        submit_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        readback_start = time.perf_counter()
        pixel_count = width * height
        left = self._read_float_buffer(self._layered_buffers[2], 3 * pixel_count).reshape(1, 3, height, width)
        right = self._read_float_buffer(self._layered_buffers[3], 3 * pixel_count).reshape(1, 3, height, width)
        mask = self._read_float_buffer(self._layered_buffers[4], pixel_count).reshape(1, 1, height, width)
        left = left.to(device=rgb.device)
        right = right.to(device=rgb.device)
        mask = mask.to(device=rgb.device)
        readback_ms = (time.perf_counter() - readback_start) * 1000.0
        debug = {
            "stereo_compute_backend": "vulkan",
            "vulkan_fused_backend": "vulkan_stereo_layered",
            "vulkan_device": self.device_name,
            "vulkan_submit_timeline": int(timeline),
            "vulkan_readback": "host_visible_storage_buffer",
            "vulkan_host_upload_ms": upload_ms,
            "vulkan_submit_wait_ms": submit_wait_ms,
            "vulkan_host_readback_ms": readback_ms,
            "vulkan_total_ms": (time.perf_counter() - total_start) * 1000.0,
        }
        return left, right, mask, debug

    @staticmethod
    def _read_float_buffer(buffer: VulkanStorageBuffer, count: int) -> torch.Tensor:
        payload = bytearray(buffer.read_bytes(int(count) * 4))
        return torch.frombuffer(payload, dtype=torch.float32).clone()

    def _close_pass_resources(self) -> None:
        if self.context is not None and not getattr(self.context, "closed", False):
            self.context.wait_idle()
        for buffer in self._buffers:
            buffer.close()
        self._buffers = ()
        if self._pass is not None:
            self._pass.close()
        self._pass = None
        self._shape = None
        for buffer in self._layered_buffers:
            buffer.close()
        self._layered_buffers = ()
        if self._layered_pass is not None:
            self._layered_pass.close()
        self._layered_pass = None
        self._layered_shape = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._close_pass_resources()
        finally:
            if self._owns_context and self.context is not None:
                self.context.close()
            self.context = None

    def __enter__(self) -> "VulkanStereoComputeBackend":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
