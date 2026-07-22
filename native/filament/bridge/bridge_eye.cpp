#include "bridge_eye.h"
#include "bridge_internal.h"

void bridge_eye_activate(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->eyes.size()) return;
    auto& eye = bridge->eyes[eye_index];
    bridge->active_eye = eye_index;
    bridge->renderer = eye.renderer;
    bridge->view = eye.view;
    bridge->camera = eye.camera;
    bridge->color_grading = eye.color_grading;
    bridge->swapchain = eye.swapchain;
    bridge->external_swapchain = eye.external_swapchain;
    bridge->frame_active = eye.frame_active;
    bridge->screen_texture = bridge->screen_textures[eye_index];
    if (bridge->screen_texture && bridge->screen_material_instance) {
        bridge->screen_material_instance->setParameter(
                "screenTexture", bridge->screen_texture,
                bridge->screen_texture_sampler);
    }
}

int bridge_eye_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height) {
    return bridge_eye_create_target_swapchain(
            bridge, 0, image_handles, image_count, format, width, height);
}

int bridge_eye_create_target_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height) {
    if (!bridge || !bridge->engine || !bridge->platform ||
            eye_index >= bridge->eyes.size()) return 0;
    auto& eye = bridge->eyes[eye_index];
    if (eye.swapchain) {
        bridge->engine->destroy(eye.swapchain);
        eye.swapchain = nullptr;
        eye.external_swapchain = nullptr;
    }
    auto* external = bridge->platform->create_external_swapchain(
            image_handles, image_count, static_cast<VkFormat>(format), width, height);
    if (!external) {
        bridge_set_error(bridge, "Invalid OpenXR Vulkan swapchain image list");
        return 0;
    }
    uint64_t swapchain_flags = 0;
    if (static_cast<VkFormat>(format) == VK_FORMAT_R8G8B8A8_SRGB ||
            static_cast<VkFormat>(format) == VK_FORMAT_B8G8R8A8_SRGB) {
        swapchain_flags = filament::SwapChain::CONFIG_SRGB_COLORSPACE;
    }
    eye.swapchain = bridge->engine->createSwapChain(external, swapchain_flags);
    if (!eye.swapchain) {
        bridge->platform->destroy(external);
        bridge_set_error(bridge, "Filament Vulkan SwapChain creation failed");
        return 0;
    }
    eye.external_swapchain =
            static_cast<OpenXrVulkanPlatform::ExternalSwapChain*>(external);
    std::fprintf(stderr,
            "[FilamentBridge] eye swapchain created eye=%u images=%u format=%d "
            "extent=%ux%u first_image=%p\n",
            eye_index, image_count, format, width, height,
            image_count ? reinterpret_cast<void*>(
                    reinterpret_cast<uintptr_t>(eye.external_swapchain->images[0]))
                         : nullptr);
    std::fflush(stderr);
    eye.camera->setProjection(
            45.0,
            static_cast<double>(width) / static_cast<double>(height),
            0.05,
            1000.0);
    eye.view->setViewport(filament::Viewport{0, 0, width, height});
    eye.laser_view->setViewport(filament::Viewport{0, 0, width, height});
    bridge_eye_activate(bridge, eye_index);
    return 1;
}

int bridge_eye_set_active(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->eyes.size()) return 0;
    if (bridge->frame_active || bridge->eyes[eye_index].frame_active) return 0;
    bridge_eye_activate(bridge, eye_index);
    return 1;
}

int bridge_eye_set_acquired_image(FilamentBridge* bridge, uint32_t image_index) {
    if (!bridge || !bridge->swapchain || !bridge->platform) return 0;
    const int result = bridge->platform->set_pending_image(
            bridge->external_swapchain, image_index) ? 1 : 0;
    if (bridge->diagnostic_frame_count < 8) {
        const auto& images = bridge->external_swapchain->images;
        const void* image = image_index < images.size()
                ? reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(images[image_index]))
                : nullptr;
        std::fprintf(stderr,
                "[FilamentBridge] acquired eye=%u index=%u image=%p result=%d\n",
                bridge->active_eye, image_index, image, result);
        std::fflush(stderr);
    }
    return result;
}

int bridge_eye_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    if (!bridge || !bridge->camera) return 0;
    bridge->camera->lookAt(
            filament::math::float3{eye_x, eye_y, eye_z},
            filament::math::float3{center_x, center_y, center_z},
            filament::math::float3{up_x, up_y, up_z});
    return 1;
}

int bridge_eye_set_camera_projection(
        FilamentBridge* bridge,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    if (!bridge || !bridge->camera || vertical_fov_degrees <= 0.0 ||
            aspect <= 0.0 || near_plane <= 0.0 || far_plane <= near_plane) {
        return 0;
    }
    bridge->camera->setProjection(
            vertical_fov_degrees, aspect, near_plane, far_plane);
    return 1;
}

int bridge_eye_set_camera_projection_frustum(
        FilamentBridge* bridge,
        double left, double right, double bottom, double top,
        double near_plane, double far_plane) {
    if (!bridge || !bridge->camera || right <= left || top <= bottom ||
            near_plane <= 0.0 || far_plane <= near_plane) {
        return 0;
    }
    bridge->camera->setProjection(
            filament::Camera::Projection::PERSPECTIVE,
            left, right, bottom, top, near_plane, far_plane);
    return 1;
}

int bridge_eye_begin_frame(FilamentBridge* bridge) {
    if (!bridge || !bridge->renderer || !bridge->swapchain || bridge->frame_active) {
        return 0;
    }
    bridge->frame_active = bridge->renderer->beginFrame(bridge->swapchain);
    bridge->eyes[bridge->active_eye].frame_active = bridge->frame_active;
    if (!bridge->frame_active) {
        bridge_set_error(bridge, "Filament Renderer::beginFrame failed");
    }
    if (bridge->diagnostic_frame_count < 8) {
        std::fprintf(stderr,
                "[FilamentBridge] begin eye=%u renderer=%p swapchain=%p active=%d\n",
                bridge->active_eye, static_cast<void*>(bridge->renderer),
                static_cast<void*>(bridge->swapchain), bridge->frame_active ? 1 : 0);
        std::fflush(stderr);
    }
    bridge->renderer->render(bridge->view);
    bridge->renderer->render(bridge->eyes[bridge->active_eye].laser_view);
    return bridge->frame_active ? 1 : 0;
}

int bridge_eye_end_frame(FilamentBridge* bridge) {
    if (!bridge || !bridge->renderer || !bridge->frame_active) return 0;
    bridge->renderer->endFrame();
    bridge->frame_active = false;
    bridge->eyes[bridge->active_eye].frame_active = false;
    if (!bridge->engine) return 0;
    // The shared Engine switches between two external Vulkan swapchains. A
    // non-blocking flush is insufficient here: the next eye may call
    // beginFrame while the backend is still consuming the previous swapchain.
    // Complete this eye before switching targets to keep the external
    // swapchain lifetime valid. This is the safe baseline until a native
    // multi-swapchain frame scheduler is added.
    bridge->engine->flushAndWait();
    if (bridge->diagnostic_frame_count < 8) {
        std::fprintf(stderr, "[FilamentBridge] end eye=%u\n", bridge->active_eye);
        std::fflush(stderr);
        if (bridge->active_eye == 1) {
            ++bridge->diagnostic_frame_count;
        }
    }
    return 1;
}

int bridge_eye_set_ready_semaphore(
        FilamentBridge* bridge, const void* semaphore) {
    if (!bridge || !bridge->platform || !bridge->swapchain || !semaphore) return 0;
    const auto ready = reinterpret_cast<VkSemaphore>(
            const_cast<void*>(semaphore));
    return bridge->platform->set_pending_ready_semaphore(bridge->external_swapchain, ready)
            ? 1 : 0;
}
