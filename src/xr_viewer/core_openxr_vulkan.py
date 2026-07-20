from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from viewer.vulkan_context import (
    MIN_VULKAN_API_VERSION,
    VulkanContext,
    VulkanCapabilityError,
    _require_timeline_semaphore_features,
    find_graphics_queue_family,
    make_vulkan_version,
)
from viewer.vulkan_resources import VulkanExportableImage, VulkanImageResource
from app_runtime.output_contract import VulkanStereoOutputFrame

from .xr_math import _xr_quat_to_mat4, euler_to_mat4, mat4_to_xr_posef


class OpenXrVulkanUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenXrVulkanConfig:
    application_name: str = "Desktop2Stereo Vulkan"
    render_scale: float = 1.0
    clear_color: tuple[float, float, float, float] = (0.02, 0.04, 0.08, 1.0)
    requested_vulkan_version: int = make_vulkan_version(1, 4, 0)
    filament_bridge_path: str | None = None
    filament_glb_path: str | None = None
    filament_profile_path: str | None = None
    filament_scene_exposure_ev: float = 0.0
    filament_skybox_brightness: float = 1.0


@dataclass(slots=True)
class _EyeSwapchain:
    handle: Any
    images: list[Any]
    width: int
    height: int
    resources: list[VulkanImageResource] = field(default_factory=list)


class OpenXrCompositionBuilder:
    """Builds projection layers without owning OpenXR frame lifecycle."""

    def __init__(self, xr: Any, reference_space: Any) -> None:
        self.xr = xr
        self.reference_space = reference_space

    def projection_layer(
        self, views: list[Any], swapchains: list[_EyeSwapchain]
    ) -> Any:
        if len(views) < len(swapchains):
            raise ValueError("projection layer requires one view per eye swapchain")
        projection_views = []
        for eye_index, eye in enumerate(swapchains):
            projection_views.append(
                self.xr.CompositionLayerProjectionView(
                    pose=views[eye_index].pose,
                    fov=views[eye_index].fov,
                    sub_image=self.xr.SwapchainSubImage(
                        swapchain=eye.handle,
                        image_rect=self.xr.Rect2Di(
                            offset=self.xr.Offset2Di(x=0, y=0),
                            extent=self.xr.Extent2Di(width=eye.width, height=eye.height),
                        ),
                        image_array_index=0,
                    ),
                )
            )
        return self.xr.CompositionLayerProjection(
            space=self.reference_space,
            views=projection_views,
        )


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
        self._profile_head_transform: np.ndarray | None = None
        self._profile_initial_head: np.ndarray | None = None
        self._profile_view_name: str | None = None
        self._profile_near_plane = 0.05
        self._profile_far_plane = 1000.0
        self._initialized = False
        self._pending_output: VulkanStereoOutputFrame | None = None

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
            self._load_filament_profile()
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

    def run_until(self, shutdown_event: Any) -> int:
        """Run the XR frame loop until the application shutdown event is set."""

        self.initialize()
        try:
            while not shutdown_event.is_set() and not self.exit_requested:
                if not self.run_frame():
                    break
            return self.frame_count
        finally:
            self.close()

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
                for resource in reversed(eye.resources):
                    try:
                        if self.vulkan is not None:
                            self.vulkan.unregister_external_image(resource)
                    except Exception:
                        pass
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
        self._pending_output = None

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
        try:
            timeline_features, synchronization2_enabled = _require_timeline_semaphore_features(
                vk, vk_physical_device
            )
        except VulkanCapabilityError as exc:
            raise OpenXrVulkanUnavailableError(str(exc)) from exc
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        runtime_extensions = tuple(
            name
            for name in str(xr.get_vulkan_device_extensions_khr(self.instance, self.system_id)).split()
            if name
        )
        external_extensions = VulkanExportableImage.required_device_extensions()
        available_extensions = {
            _decode_name(item.extensionName)
            for item in vk.vkEnumerateDeviceExtensionProperties(vk_physical_device, None)
        }
        missing_extensions = [
            name for name in external_extensions if name not in available_extensions
        ]
        if missing_extensions:
            raise OpenXrVulkanUnavailableError(
                "Vulkan external-memory extensions are unavailable: "
                + ", ".join(missing_extensions)
            )
        device_extensions = tuple(dict.fromkeys((*runtime_extensions, *external_extensions)))
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=timeline_features,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=list(device_extensions),
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
            timeline_semaphore_enabled=True,
            synchronization2_enabled=synchronization2_enabled,
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
                _EyeSwapchain(
                    handle=handle,
                    images=images,
                    width=width,
                    height=height,
                    resources=self._register_swapchain_images(images, width, height),
                )
            )

    def _register_swapchain_images(
        self, images: list[Any], width: int, height: int
    ) -> list[VulkanImageResource]:
        resources: list[VulkanImageResource] = []
        try:
            for index, item in enumerate(images):
                image = self.vulkan.image_handle_from_address(
                    _ctypes_handle_address(item.image)
                )
                resource = VulkanImageResource(
                    context=self.vulkan,
                    image=image,
                    view=None,
                    width=width,
                    height=height,
                    format=int(self.swapchain_format),
                    layout=self.vulkan.vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    access_mask=0,
                    stage_mask=0,
                    queue_family_index=self.vulkan.queue_family_index,
                    external=True,
                    label=f"openxr-swapchain-{index}",
                )
                self.vulkan.register_external_image(resource)
                resources.append(resource)
        except Exception:
            for resource in reversed(resources):
                try:
                    self.vulkan.unregister_external_image(resource)
                except Exception:
                    pass
            raise
        return resources

    def submit_output(self, frame: VulkanStereoOutputFrame) -> None:
        """Queue the newest Vulkan left/right frame for the next XR frame."""

        if not isinstance(frame.left_eye, VulkanImageResource) or not isinstance(
            frame.right_eye, VulkanImageResource
        ):
            raise TypeError("OpenXR Vulkan output requires VulkanImageResource eyes")
        if frame.left_eye.context is not self.vulkan or frame.right_eye.context is not self.vulkan:
            raise ValueError("OpenXR output images belong to a different Vulkan context")
        self._pending_output = frame

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
                    bridge.set_scene_exposure(self.config.filament_scene_exposure_ev)
                    bridge.set_skybox_brightness(self.config.filament_skybox_brightness)
                    self.filament_bridges.append(bridge)
                except Exception:
                    bridge.close()
                    raise
        except Exception:
            for bridge in reversed(self.filament_bridges):
                bridge.close()
            self.filament_bridges.clear()
            raise

    def _load_filament_profile(self) -> None:
        profile_path = self.config.filament_profile_path
        if not profile_path:
            return
        with open(profile_path, "r", encoding="utf-8-sig") as handle:
            profile = json.load(handle)
        if not isinstance(profile, dict):
            raise ValueError("Filament profile root must be an object")

        view_pose = profile.get("view_pose", profile.get("camera"))
        view_poses = profile.get("view_poses")
        if isinstance(view_poses, list) and view_poses:
            index = int(profile.get("view_pose_index", 0)) % len(view_poses)
            view_pose = view_poses[index]
        if not isinstance(view_pose, dict):
            raise ValueError("Filament profile does not contain a view pose")

        try:
            model_position = profile.get("model_position", profile.get("position", [0.0, 0.0, 0.0]))
            if not isinstance(model_position, (list, tuple)) or len(model_position) < 3:
                model_position = [0.0, 0.0, 0.0]
            model_rotation_deg = profile.get("model_rotation_deg", [0.0, 0.0, 0.0])
            if not isinstance(model_rotation_deg, (list, tuple)) or len(model_rotation_deg) < 3:
                model_rotation_deg = [0.0, 0.0, 0.0]
            model_scale = profile.get("model_scale", [1.0, 1.0, 1.0])
            if not isinstance(model_scale, (list, tuple)) or len(model_scale) < 3:
                model_scale = [1.0, 1.0, 1.0]

            world_position_vec = np.asarray(
                [float(view_pose[key]) for key in ("x", "y", "z")],
                dtype=np.float32,
            )
            rotation_deg = view_pose.get("rotation_deg")
            if not isinstance(rotation_deg, (list, tuple)) or len(rotation_deg) < 3:
                rotation_deg = [float(view_pose.get("angle", 0.0)), 0.0, 0.0]
            rotation_rad = [math.radians(float(value)) for value in rotation_deg[:3]]

            model_matrix = euler_to_mat4(
                *(math.radians(float(value)) for value in model_rotation_deg[:3])
            ).astype(np.float32)
            model_matrix[:3, 3] = np.asarray(model_position[:3], dtype=np.float32)
            scale = np.asarray(model_scale[:3], dtype=np.float32)
            model_matrix[:3, :3] = model_matrix[:3, :3] @ np.diag(scale)
            position = (np.linalg.inv(model_matrix) @ np.append(world_position_vec, 1.0))[:3]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Filament profile view pose contains invalid values") from exc

        transform = euler_to_mat4(*rotation_rad).astype(np.float32)
        transform[:3, 3] = np.asarray(position, dtype=np.float32)
        self._profile_head_transform = transform
        self._profile_view_name = str(view_pose.get("name", "profile"))
        self._profile_near_plane = max(0.001, float(profile.get("xr_projection_near", 0.05)))
        self._profile_far_plane = max(
            self._profile_near_plane + 1.0,
            float(profile.get("xr_projection_far", 1000.0)),
        )
        print(
            f"Loaded Filament profile view: {self._profile_view_name} "
            f"world_position={world_position_vec.tolist()} glb_position={position.tolist()} "
            f"rotation_rad={rotation_rad}",
            flush=True,
        )

    def _apply_filament_profile(self, views: list[Any]) -> list[Any]:
        if self._profile_head_transform is None or len(views) < 2:
            return views
        eye_matrices = [_xr_view_pose_to_model_mat4(view.pose) for view in views[:2]]
        if self._profile_initial_head is None:
            initial_head = eye_matrices[0].copy()
            initial_head[:3, 3] = (eye_matrices[0][:3, 3] + eye_matrices[1][:3, 3]) * 0.5
            self._profile_initial_head = initial_head
            print("Filament profile tracking anchor captured.", flush=True)

        tracking_delta = self._profile_head_transform @ np.linalg.inv(self._profile_initial_head)
        adjusted = []
        for view, eye_matrix in zip(views, eye_matrices):
            adjusted_pose = mat4_to_xr_posef(tracking_delta @ eye_matrix)
            adjusted.append(SimpleNamespace(pose=adjusted_pose, fov=view.fov))
        return adjusted

    def _render_projection_layer(self, views: list[Any]) -> Any | None:
        if len(views) < len(self.swapchains):
            return None
        views = self._apply_filament_profile(views)
        xr = self.xr
        output_frame = self._pending_output
        for eye_index, eye in enumerate(self.swapchains):
            with _acquired_swapchain_image(xr, eye) as image_index:
                if eye_index < len(self.filament_bridges):
                    bridge = self.filament_bridges[eye_index]
                    _update_filament_camera(
                        bridge,
                        views[eye_index],
                        near_plane=self._profile_near_plane,
                        far_plane=self._profile_far_plane,
                    )
                    bridge.set_acquired_image(image_index)
                    bridge.begin_frame()
                    bridge.end_frame()
                else:
                    if output_frame is not None:
                        source = (
                            output_frame.left_eye
                            if eye_index == 0
                            else output_frame.right_eye
                        )
                        self.vulkan.copy_image(
                            source,
                            eye.resources[image_index],
                            wait_for_timeline=output_frame.ready_timeline,
                        )
                    else:
                        image_address = _ctypes_handle_address(eye.images[image_index].image)
                        image = self.vulkan.image_handle_from_address(image_address)
                        self.vulkan.clear_color_image(image, self.config.clear_color)
        self._pending_output = None
        return OpenXrCompositionBuilder(xr, self.reference_space).projection_layer(
            views, self.swapchains
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("OpenXrVulkanPresenter is not initialized")


@contextmanager
def _acquired_swapchain_image(xr: Any, eye: _EyeSwapchain):
    """Guarantee release after every successful acquire, including wait errors."""

    image_index = xr.acquire_swapchain_image(eye.handle)
    try:
        xr.wait_swapchain_image(
            eye.handle,
            xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
        )
        yield image_index
    finally:
        xr.release_swapchain_image(eye.handle)


def _xr_view_pose_to_model_mat4(pose: Any) -> np.ndarray:
    matrix = _xr_quat_to_mat4(pose.orientation).astype(np.float32)
    matrix[:3, 3] = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    return matrix


def _update_filament_camera(
    bridge: Any,
    view: Any,
    *,
    near_plane: float = 0.05,
    far_plane: float = 1000.0,
) -> None:
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
    left = math.tan(float(fov.angle_left)) * near_plane
    right = math.tan(float(fov.angle_right)) * near_plane
    bottom = math.tan(float(fov.angle_down)) * near_plane
    top = math.tan(float(fov.angle_up)) * near_plane
    if hasattr(bridge, "set_camera_projection_frustum"):
        bridge.set_camera_projection_frustum(
            left, right, bottom, top,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        return
    horizontal = max(0.01, abs(float(fov.angle_right) - float(fov.angle_left)))
    vertical = max(0.01, abs(float(fov.angle_up) - float(fov.angle_down)))
    aspect = math.tan(horizontal * 0.5) / max(math.tan(vertical * 0.5), 1e-6)
    bridge.set_camera_projection(
        math.degrees(vertical),
        aspect,
        near_plane=near_plane,
        far_plane=far_plane,
    )


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
    if maximum < MIN_VULKAN_API_VERSION:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime does not support the required Vulkan 1.2 minimum"
        )
    selected = max(minimum, min(int(requested), maximum))
    if selected < MIN_VULKAN_API_VERSION:
        raise OpenXrVulkanUnavailableError(
            "Negotiated Vulkan API version is below the required Vulkan 1.2 minimum"
        )
    return selected


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
