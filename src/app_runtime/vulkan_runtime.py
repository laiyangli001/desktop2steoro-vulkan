from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from stereo_runtime.vulkan_image_pass import VulkanImageCopyPass
from viewer.vulkan_context import VulkanCapabilityError, VulkanContext
from viewer.vulkan_descriptors import VulkanStorageImage
from viewer.vulkan_resources import VulkanImageResource


@dataclass(frozen=True, slots=True)
class VulkanRuntimeConfig:
    width: int
    height: int
    shader_path: str | Path = "shaders/d2s_copy_image.spv"

    def __post_init__(self) -> None:
        if int(self.width) < 1 or int(self.height) < 1:
            raise ValueError("Vulkan runtime dimensions must be positive")


class VulkanDeviceLostError(VulkanCapabilityError):
    pass


class VulkanRuntimeSession:
    """Owns Vulkan graph resources for GPU-image runtime submissions."""

    def __init__(
        self,
        context: VulkanContext,
        config: VulkanRuntimeConfig,
        *,
        owns_context: bool = False,
    ) -> None:
        self.context = context
        self.config = config
        self._owns_context = owns_context
        self._device_lost = False
        self._last_error: str | None = None
        self.image_copy_pass: VulkanImageCopyPass | None = None
        try:
            self.image_copy_pass = VulkanImageCopyPass(
                context,
                width=config.width,
                height=config.height,
                shader_path=config.shader_path,
            )
        except Exception:
            self.close()
            raise

    @classmethod
    def create(cls, config: VulkanRuntimeConfig) -> "VulkanRuntimeSession":
        context = VulkanContext.create()
        try:
            return cls(context, config, owns_context=True)
        except Exception:
            context.close()
            raise

    def submit_image_pair(
        self,
        source_image: VulkanStorageImage | VulkanImageResource,
        output_image: VulkanStorageImage | VulkanImageResource,
        *,
        frame_id: int,
        config_version: int,
        ready_timeline: int | None = None,
    ) -> int | None:
        if self.image_copy_pass is None:
            raise RuntimeError("Vulkan runtime session is closed")
        self._ensure_healthy()
        try:
            return self.image_copy_pass.submit(
                source_image,
                output_image,
                frame_id=frame_id,
                config_version=config_version,
                ready_timeline=ready_timeline,
            )
        except Exception as exc:
            if _is_device_lost(exc):
                self._device_lost = True
                self._last_error = str(exc) or type(exc).__name__
                raise VulkanDeviceLostError(
                    "Vulkan device was lost; recreate the runtime session"
                ) from exc
            raise

    def submit_external_image_pair(
        self,
        source_image: VulkanImageResource,
        output_image: VulkanImageResource,
        *,
        frame_id: int,
        config_version: int,
        ready_timeline: int | None = None,
    ) -> int | None:
        """Submit producer-owned images without a CPU readback or handle copy."""

        for image in (source_image, output_image):
            if image.context is not self.context:
                raise ValueError("external image belongs to a different Vulkan context")
            if not image.external:
                raise ValueError("submit_external_image_pair requires external images")
        return self.submit_image_pair(
            source_image,
            output_image,
            frame_id=frame_id,
            config_version=config_version,
            ready_timeline=ready_timeline,
        )

    @property
    def device_lost(self) -> bool:
        return self._device_lost

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def wait_idle(self) -> None:
        self.context.wait_idle()

    def resize(self, width: int, height: int) -> None:
        if self.image_copy_pass is None:
            raise RuntimeError("Vulkan runtime session is closed")
        self._ensure_healthy()
        next_config = VulkanRuntimeConfig(
            width=width,
            height=height,
            shader_path=self.config.shader_path,
        )
        next_pass = VulkanImageCopyPass(
            self.context,
            width=next_config.width,
            height=next_config.height,
            shader_path=next_config.shader_path,
        )
        try:
            self.context.wait_idle()
        except Exception as exc:
            next_pass.close()
            if _is_device_lost(exc):
                self._device_lost = True
                self._last_error = str(exc) or type(exc).__name__
            raise
        previous_pass = self.image_copy_pass
        self.image_copy_pass = next_pass
        self.config = next_config
        previous_pass.close()

    def close(self) -> None:
        if self.image_copy_pass is None and not self._owns_context:
            return
        try:
            self.context.wait_idle()
        finally:
            if self.image_copy_pass is not None:
                self.image_copy_pass.close()
            self.image_copy_pass = None
            if self._owns_context:
                self.context.close()
                self._owns_context = False

    def __enter__(self) -> "VulkanRuntimeSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ensure_healthy(self) -> None:
        if self._device_lost:
            raise VulkanDeviceLostError(
                "Vulkan device was lost; recreate the runtime session"
            )


def _is_device_lost(exc: BaseException) -> bool:
    marker = f"{type(exc).__name__} {exc}".lower().replace("_", " ")
    return "device lost" in marker or "error device lost" in marker
