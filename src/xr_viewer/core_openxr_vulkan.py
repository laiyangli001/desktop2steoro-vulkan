from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from viewer.vulkan_context import VulkanContext, find_graphics_queue_family, make_vulkan_version

from .xr_math import _xr_quat_to_mat4


class OpenXrVulkanUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenXrVulkanConfig:
    application_name: str = "Desktop2Stereo Vulkan"
    render_scale: float = 1.0
    clear_color: tuple[float, float, float, float] = (0.02, 0.04, 0.08, 1.0)
    requested_vulkan_version: int = make_vulkan_version(1, 2, 0)
    filament_bridge_path: str | None = None
    filament_glb_path: str | None = None


@dataclass(slots=True)
class _EyeSwapchain:
    handle: Any
    images: list[Any]
    width: int
    height: int


class OpenXrVulkanPresenter:
    """Minimal OpenXR Vulkan projection-layer presenter."""

    _VULKAN_EXTENSION = "XR_KHR_vulkan_enable2"

    def __init__(self, config: OpenXrVulkanConfig | None = None) -> None:
        self.config = config or OpenXrVulkanConfig()
        if self.config.render_scale <= 0:
            raise ValueError("render_scale must be greater than zero")
        if len(self.config.clear_color) != 4:
            raise ValueError("clear_color must contain four components")

        self.xr: Any = None
        self.instance: Any = None
        self.system_id: Any = None
        self.session: Any = None
        self.reference_space: Any = None
        self.vulkan: VulkanContext | None = None
        self.swapchain_format: int | None = None
        self.swapchains: list[_EyeSwapchain] = []
        self.filament_bridges: list[Any] = []
        self.session_state: Any = None
        self.session_running = False
        self.exit_requested = False
        self.frame_count = 0
        self._view_configuration_type: Any = None
        self._environment_blend_mode: Any = None
        self._vulkan_loader: Any = None
        self._vk_get_instance_proc_addr: Any = None
        self._graphics_binding: Any = None
        self._provisional_vk_instance: Any = None
        self._provisional_vk_device: Any = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        self.exit_requested = False
        self.frame_count = 0
        self.session_state = None
        self.xr = _import_openxr()
        xr = self.xr
        available_extensions = {
            _decode_name(item.extension_name)
            for item in xr.enumerate_instance_extension_properties()
        }
        if self._VULKAN_EXTENSION not in available_extensions:
            raise OpenXrVulkanUnavailableError(
                f"OpenXR runtime does not expose {self._VULKAN_EXTENSION}"
            )

        try:
            self.instance = xr.create_instance(
                xr.InstanceCreateInfo(
                    application_info=xr.ApplicationInfo(
                        application_name=self.config.application_name,
                        application_version=1,
                        engine_name="D2S",
                        engine_version=1,
                        api_version=xr.Version(1, 0, 0),
                    ),
                    enabled_extension_names=[self._VULKAN_EXTENSION],
                )
            )
            self.system_id = xr.get_system(
                self.instance,
                xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
            )
            requirements = _get_vulkan_graphics_requirements2(
                xr, self.instance, self.system_id
            )
            api_version = _select_vulkan_api_version(
                requirements, self.config.requested_vulkan_version
            )
            self._create_vulkan_objects(api_version)
            self._create_session_and_swapchains()
            self._initialize_filament_bridges()
            self._initialized = True
        except Exception:
            self.close()
            raise

    def poll_events(self) -> None:
        self._ensure_initialized()
        xr = self.xr
        while True:
            try:
                event = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return

            if event.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                changed = ctypes.cast(
                    ctypes.byref(event),
                    ctypes.POINTER(xr.EventDataSessionStateChanged),
                ).contents
                self.session_state = changed.state
                if changed.state == xr.SessionState.READY and not self.session_running:
                    xr.begin_session(
                        self.session,
                        xr.SessionBeginInfo(
                            primary_view_configuration_type=self._view_configuration_type
                        ),
                    )
                    self.session_running = True
                elif changed.state == xr.SessionState.STOPPING and self.session_running:
                    xr.end_session(self.session)
                    self.session_running = False
                elif changed.state in (
                    xr.SessionState.EXITING,
                    xr.SessionState.LOSS_PENDING,
                ):
                    self.exit_requested = True
            elif event.type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                self.exit_requested = True

    def run_frame(self) -> bool:
        self._ensure_initialized()
        self.poll_events()
        if self.exit_requested:
            return False
        if not self.session_running:
            time.sleep(0.01)
            return True

        xr = self.xr
        frame_state = xr.wait_frame(self.session)
        xr.begin_frame(self.session)
        layer_structures: list[Any] = []
        layer_pointers: list[Any] = []
        try:
            if frame_state.should_render:
                view_state, views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(
                        view_configuration_type=self._view_configuration_type,
                        display_time=frame_state.predicted_display_time,
                        space=self.reference_space,
                    ),
                )
                valid_flags = (
                    xr.ViewStateFlags.POSITION_VALID_BIT
                    | xr.ViewStateFlags.ORIENTATION_VALID_BIT
                )
                if view_state.view_state_flags & valid_flags == valid_flags:
                    layer = self._render_projection_layer(views)
                    if layer is not None:
                        layer_structures.append(layer)
                        layer_pointers.append(ctypes.pointer(layer))
        finally:
            end_info = xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=self._environment_blend_mode,
                layer_count=len(layer_pointers),
                layers=layer_pointers or None,
            )
            xr.end_frame(self.session, end_info)
        self.frame_count += 1
        return not self.exit_requested

    def run(self, frame_limit: int | None = None) -> int:
        self.initialize()
        while frame_limit is None or self.frame_count < frame_limit:
            if not self.run_frame():
                break
        return self.frame_count

    def close(self) -> None:
        xr = self.xr
        if self.vulkan is not None:
            try:
                self.vulkan.wait_idle()
            except Exception:
                pass

        for bridge in reversed(self.filament_bridges):
            try:
                bridge.close()
            except Exception:
                pass
        self.filament_bridges.clear()

        if xr is not None:
            for eye in reversed(self.swapchains):
                try:
                    xr.destroy_swapchain(eye.handle)
                except Exception:
                    pass
            self.swapchains.clear()

            if self.reference_space is not None:
                try:
                    xr.destroy_space(self.reference_space)
                except Exception:
                    pass
                self.reference_space = None

            if self.session is not None:
                if self.session_running:
                    try:
                        xr.end_session(self.session)
                    except Exception:
                        pass
                try:
                    xr.destroy_session(self.session)
                except Exception:
                    pass
                self.session = None
                self.session_running = False

        if self.vulkan is not None:
            try:
                self.vulkan.close()
            except Exception:
                pass
            self.vulkan = None
        elif self._provisional_vk_instance is not None:
            try:
                import vulkan as vk

                if self._provisional_vk_device is not None:
                    vk.vkDestroyDevice(self._provisional_vk_device, None)
                vk.vkDestroyInstance(self._provisional_vk_instance, None)
            except Exception:
                pass
        self._provisional_vk_device = None
        self._provisional_vk_instance = None

        if xr is not None and self.instance is not None:
            try:
                xr.destroy_instance(self.instance)
            except Exception:
                pass
            self.instance = None

        self.system_id = None
        self.swapchain_format = None
        self._graphics_binding = None
        self._initialized = False

    def __enter__(self) -> "OpenXrVulkanPresenter":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _create_vulkan_objects(self, api_version: int) -> None:
        xr = self.xr
        import vulkan as vk

        self._vulkan_loader, self._vk_get_instance_proc_addr = _load_vulkan_proc_addr(xr)
        platform = _openxr_platform_module(xr)

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.config.application_name,
            applicationVersion=1,
            pEngineName="D2S",
            engineVersion=1,
            apiVersion=int(api_version),
        )
        instance_create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        xr_instance, vulkan_result = xr.create_vulkan_instance_khr(
            self.instance,
            xr.VulkanInstanceCreateInfoKHR(
                system_id=self.system_id,
                pfn_get_instance_proc_addr=self._vk_get_instance_proc_addr,
                vulkan_create_info=_cffi_struct_pointer(
                    vk, instance_create_info, platform.VkInstanceCreateInfo
                ),
            ),
        )
        _check_vulkan_result(vulkan_result, "xrCreateVulkanInstanceKHR")
        vk_instance = _ctypes_handle_to_cffi(vk, "VkInstance", xr_instance)
        self._provisional_vk_instance = vk_instance

        xr_physical_device = xr.get_vulkan_graphics_device2_khr(
            self.instance,
            xr.VulkanGraphicsDeviceGetInfoKHR(
                system_id=self.system_id,
                vulkan_instance=xr_instance,
            ),
        )
        vk_physical_device = _ctypes_handle_to_cffi(
            vk, "VkPhysicalDevice", xr_physical_device
        )
        queue_family_index = find_graphics_queue_family(vk, vk_physical_device)
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
        )
        xr_device, vulkan_result = xr.create_vulkan_device_khr(
            self.instance,
            xr.VulkanDeviceCreateInfoKHR(
                system_id=self.system_id,
                pfn_get_instance_proc_addr=self._vk_get_instance_proc_addr,
                vulkan_physical_device=xr_physical_device,
                vulkan_create_info=_cffi_struct_pointer(
                    vk, device_create_info, platform.VkDeviceCreateInfo
                ),
            ),
        )
        _check_vulkan_result(vulkan_result, "xrCreateVulkanDeviceKHR")
        vk_device = _ctypes_handle_to_cffi(vk, "VkDevice", xr_device)
        self._provisional_vk_device = vk_device
        self.vulkan = VulkanContext.adopt(
            instance=vk_instance,
            physical_device=vk_physical_device,
            device=vk_device,
            queue_family_index=queue_family_index,
            owns_instance=True,
            owns_device=True,
        )
        self._provisional_vk_device = None
        self._provisional_vk_instance = None
        self._graphics_binding = xr.GraphicsBindingVulkan2KHR(
            instance=xr_instance,
            physical_device=xr_physical_device,
            device=xr_device,
            queue_family_index=queue_family_index,
            queue_index=0,
        )

    def _create_session_and_swapchains(self) -> None:
        xr = self.xr
        vk = self.vulkan.vk
        self._view_configuration_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self._environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE
        self.session = xr.create_session(
            self.instance,
            xr.SessionCreateInfo(
                system_id=self.system_id,
                next=ctypes.cast(
                    ctypes.pointer(self._graphics_binding), ctypes.c_void_p
                ),
            ),
        )
        self.reference_space = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL
            ),
        )
        formats = list(xr.enumerate_swapchain_formats(self.session))
        self.swapchain_format = _select_swapchain_format(vk, formats)
        view_configs = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self._view_configuration_type
        )
        if len(view_configs) < 2:
            raise OpenXrVulkanUnavailableError(
                f"PRIMARY_STEREO returned {len(view_configs)} view(s)"
            )

        for view_config in view_configs[:2]:
            width = _scaled_dimension(
                view_config.recommended_image_rect_width,
                view_config.max_image_rect_width,
                self.config.render_scale,
            )
            height = _scaled_dimension(
                view_config.recommended_image_rect_height,
                view_config.max_image_rect_height,
                self.config.render_scale,
            )
            handle = xr.create_swapchain(
                self.session,
                xr.SwapchainCreateInfo(
                    usage_flags=(
                        xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                        | xr.SwapchainUsageFlags.TRANSFER_DST_BIT
                    ),
                    format=self.swapchain_format,
                    sample_count=1,
                    width=width,
                    height=height,
                    face_count=1,
                    array_size=1,
                    mip_count=1,
                ),
            )
            images = list(
                xr.enumerate_swapchain_images(handle, xr.SwapchainImageVulkan2KHR)
            )
            if not images:
                xr.destroy_swapchain(handle)
                raise OpenXrVulkanUnavailableError(
                    "OpenXR runtime returned an empty Vulkan swapchain"
                )
            self.swapchains.append(
                _EyeSwapchain(handle=handle, images=images, width=width, height=height)
            )

    def _initialize_filament_bridges(self) -> None:
        bridge_path = self.config.filament_bridge_path or os.environ.get(
            "D2S_FILAMENT_BRIDGE"
        )
        if not bridge_path:
            return

        from .filament_vulkan_bridge import FilamentVulkanBridge

        try:
            for eye in self.swapchains:
                bridge = FilamentVulkanBridge(bridge_path)
                try:
                    bridge.create(
                        instance=self.vulkan.instance,
                        physical_device=self.vulkan.physical_device,
                        device=self.vulkan.device,
                        queue_family_index=self.vulkan.queue_family_index,
                        queue_index=0,
                    )
                    bridge.create_swapchain(
                        (image.image for image in eye.images),
                        format=self.swapchain_format,
                        width=eye.width,
                        height=eye.height,
                    )
                    glb_path = self.config.filament_glb_path
                    if glb_path:
                        bridge.load_glb(Path(glb_path).read_bytes())
                    self.filament_bridges.append(bridge)
                except Exception:
                    bridge.close()
                    raise
        except Exception:
            for bridge in reversed(self.filament_bridges):
                bridge.close()
            self.filament_bridges.clear()
            raise

    def _render_projection_layer(self, views: list[Any]) -> Any | None:
        if len(views) < len(self.swapchains):
            return None
        xr = self.xr
        projection_views: list[Any] = []
        for eye_index, eye in enumerate(self.swapchains):
            image_index = xr.acquire_swapchain_image(eye.handle)
            image_ready = False
            try:
                xr.wait_swapchain_image(
                    eye.handle,
                    xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
                image_ready = True
                if eye_index < len(self.filament_bridges):
                    bridge = self.filament_bridges[eye_index]
                    _update_filament_camera(bridge, views[eye_index])
                    bridge.set_acquired_image(image_index)
                    bridge.begin_frame()
                    bridge.end_frame()
                else:
                    image_address = _ctypes_handle_address(eye.images[image_index].image)
                    image = self.vulkan.image_handle_from_address(image_address)
                    self.vulkan.clear_color_image(image, self.config.clear_color)
            finally:
                if image_ready:
                    xr.release_swapchain_image(eye.handle)

            projection_views.append(
                xr.CompositionLayerProjectionView(
                    pose=views[eye_index].pose,
                    fov=views[eye_index].fov,
                    sub_image=xr.SwapchainSubImage(
                        swapchain=eye.handle,
                        image_rect=xr.Rect2Di(
                            offset=xr.Offset2Di(x=0, y=0),
                            extent=xr.Extent2Di(width=eye.width, height=eye.height),
                        ),
                        image_array_index=0,
                    ),
                )
            )
        return xr.CompositionLayerProjection(
            space=self.reference_space,
            views=projection_views,
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("OpenXrVulkanPresenter is not initialized")


def _update_filament_camera(bridge: Any, view: Any) -> None:
    pose = view.pose
    rotation = _xr_quat_to_mat4(pose.orientation)[:3, :3]
    position = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    forward = rotation @ (0.0, 0.0, -1.0)
    up = rotation @ (0.0, 1.0, 0.0)
    center = tuple(position[index] + float(forward[index]) for index in range(3))
    bridge.set_camera_look_at(position, center, tuple(float(value) for value in up))

    fov = view.fov
    horizontal = max(0.01, abs(float(fov.angle_right) - float(fov.angle_left)))
    vertical = max(0.01, abs(float(fov.angle_up) - float(fov.angle_down)))
    aspect = math.tan(horizontal * 0.5) / max(math.tan(vertical * 0.5), 1e-6)
    bridge.set_camera_projection(math.degrees(vertical), aspect)


def _import_openxr() -> Any:
    try:
        import xr
    except (ImportError, OSError) as exc:
        raise OpenXrVulkanUnavailableError(
            "pyopenxr or the OpenXR loader is unavailable"
        ) from exc
    return xr


def _get_vulkan_graphics_requirements2(
    xr: Any, instance: Any, system_id: Any
) -> Any:
    function = ctypes.cast(
        xr.get_instance_proc_addr(
            instance.instance, "xrGetVulkanGraphicsRequirements2KHR"
        ),
        xr.platform.PFN_xrGetVulkanGraphicsRequirements2KHR,
    )
    requirements = xr.GraphicsRequirementsVulkan2KHR()
    result = xr.check_result(function(instance, system_id, ctypes.byref(requirements)))
    if result.is_exception():
        raise result
    return requirements


def _select_vulkan_api_version(requirements: Any, requested: int) -> int:
    minimum = make_vulkan_version(
        requirements.min_api_version_supported.major,
        requirements.min_api_version_supported.minor,
        requirements.min_api_version_supported.patch,
    )
    maximum = make_vulkan_version(
        requirements.max_api_version_supported.major,
        requirements.max_api_version_supported.minor,
        requirements.max_api_version_supported.patch,
    )
    if minimum > maximum:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime returned an invalid Vulkan API version range"
        )
    return max(minimum, min(int(requested), maximum))


def _select_swapchain_format(vk: Any, available_formats: list[int]) -> int:
    preferred = (
        vk.VK_FORMAT_R8G8B8A8_SRGB,
        vk.VK_FORMAT_B8G8R8A8_SRGB,
        vk.VK_FORMAT_R8G8B8A8_UNORM,
        vk.VK_FORMAT_B8G8R8A8_UNORM,
    )
    for candidate in preferred:
        if int(candidate) in available_formats:
            return int(candidate)
    if not available_formats:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime returned no swapchain formats"
        )
    return int(available_formats[0])


def _scaled_dimension(recommended: int, maximum: int, scale: float) -> int:
    return max(1, min(int(maximum), round(int(recommended) * float(scale))))


def _openxr_platform_module(xr: Any) -> Any:
    return importlib.import_module(xr.VulkanInstanceCreateInfoKHR.__module__)


def _load_vulkan_proc_addr(xr: Any) -> tuple[Any, Any]:
    if sys.platform == "win32":
        candidates = ["vulkan-1.dll"]
    elif sys.platform == "darwin":
        candidates = ["libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"]
    else:
        candidates = ["libvulkan.so.1", "libvulkan.so"]
    discovered = ctypes.util.find_library("vulkan")
    if discovered:
        candidates.append(discovered)

    platform = _openxr_platform_module(xr)
    errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            loader = (
                ctypes.WinDLL(candidate)
                if sys.platform == "win32"
                else ctypes.CDLL(candidate)
            )
            function = ctypes.cast(
                loader.vkGetInstanceProcAddr, platform.PFN_vkGetInstanceProcAddr
            )
            return loader, function
        except (AttributeError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise OpenXrVulkanUnavailableError(
        "Unable to load vkGetInstanceProcAddr: " + "; ".join(errors)
    )


def _cffi_struct_pointer(vk: Any, value: Any, ctypes_type: Any) -> Any:
    address = int(vk.ffi.cast("uintptr_t", vk.ffi.addressof(value)))
    return ctypes.cast(ctypes.c_void_p(address), ctypes.POINTER(ctypes_type))


def _ctypes_handle_to_cffi(vk: Any, type_name: str, handle: Any) -> Any:
    address = _ctypes_handle_address(handle)
    if not address:
        raise OpenXrVulkanUnavailableError(f"OpenXR returned a null {type_name}")
    return vk.ffi.cast(type_name, address)


def _ctypes_handle_address(handle: Any) -> int:
    return int(ctypes.cast(handle, ctypes.c_void_p).value or 0)


def _check_vulkan_result(result: Any, operation: str) -> None:
    value = int(result.value if hasattr(result, "value") else result)
    if value != 0:
        raise OpenXrVulkanUnavailableError(f"{operation} returned VkResult {value}")


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(value)
