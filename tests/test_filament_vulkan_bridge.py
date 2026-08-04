from __future__ import annotations

import ctypes
import json
import re
import struct
import subprocess
import threading
from pathlib import Path

import pytest

from xr_viewer.filament_vulkan_bridge import (
    FilamentBridgeError,
    FilamentVulkanBridge,
    _VulkanCreateInfo,
    _as_pointer_value,
    default_bridge_path,
)


def test_vulkan_create_info_has_stable_c_layout() -> None:
    assert ctypes.sizeof(_VulkanCreateInfo) == ctypes.sizeof(ctypes.c_void_p) * 3 + 8


def test_default_bridge_path_matches_platform() -> None:
    path = default_bridge_path()
    assert path.parent.name in {"windows", "linux", "macos"}
    assert path.name.startswith(("filament_bridge", "libfilament_bridge"))


def test_remote_filament_build_enables_multiview_without_stale_sdk_cache() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/filament-bridge.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("-DFILAMENT_ENABLE_MULTIVIEW=ON") == 2
    assert workflow.count(
        "hashFiles('native/filament/version.json', "
        "'native/filament/patches/**', '.github/workflows/filament-bridge.yml')"
    ) == 2


def test_pointer_value_accepts_integer_and_c_void_p() -> None:
    assert _as_pointer_value(17) == 17
    assert _as_pointer_value(ctypes.c_void_p(23)) == 23


def test_missing_bridge_library_is_reported() -> None:
    with pytest.raises(FilamentBridgeError, match="unable to load"):
        FilamentVulkanBridge("missing-filament-bridge.dll")


def test_bridge_rejects_c_abi_calls_from_non_owner_thread() -> None:
    bridge = object.__new__(FilamentVulkanBridge)
    bridge._handle = ctypes.c_void_p(1)
    bridge._owner_thread_id = threading.get_ident() + 1

    with pytest.raises(FilamentBridgeError, match="Presenter owner thread"):
        bridge._ensure_loaded()


def test_native_bridge_keeps_modular_resource_lifetimes_explicit() -> None:
    bridge_dir = (
        Path(__file__).resolve().parents[1] / "native/filament/bridge"
    )
    module_names = (
        "bridge_internal.h",
        "bridge_context.cpp",
        "bridge_context.h",
        "bridge_eye.cpp",
        "bridge_eye.h",
        "bridge_scene.cpp",
        "bridge_scene.h",
        "bridge_controller.cpp",
        "bridge_controller.h",
        "bridge_controller_guide.cpp",
        "bridge_controller_guide.h",
        "bridge_laser.cpp",
        "bridge_laser.h",
        "bridge_glow.cpp",
        "bridge_glow.h",
        "bridge_screen.cpp",
        "bridge_screen.h",
        "bridge_material.cpp",
        "bridge_material.h",
        "preview_bridge.cpp",
        "preview_bridge.h",
    )
    facade = (bridge_dir / "filament_bridge.cpp").read_text(encoding="utf-8")
    controller_source = (bridge_dir / "bridge_controller.cpp").read_text(
        encoding="utf-8"
    )
    source = facade + "\n" + "\n".join(
        (bridge_dir / name).read_text(encoding="utf-8")
        for name in module_names
    )
    cmake = (bridge_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    public_header = (bridge_dir / "filament_bridge.h").read_text(
        encoding="utf-8"
    )
    abi_pattern = re.compile(
        r"\b(filament_(?:bridge|preview)_[a-z0-9_]+)\s*\("
    )

    assert set(abi_pattern.findall(public_header)) == set(
        abi_pattern.findall(facade)
    )
    assert all(
        not abi_pattern.search((bridge_dir / name).read_text(encoding="utf-8"))
        for name in module_names
    )
    assert "filament::" not in facade
    assert len(facade.splitlines()) < 430
    assert "filament_bridge_set_screen_source_version" in facade
    assert all(name in cmake for name in module_names if name.endswith(".cpp"))
    assert "filament::Renderer* renderer = nullptr;" in source
    assert "auto* shared_renderer = bridge->engine->createRenderer();" in source
    assert "eye.renderer = shared_renderer;" in source
    assert "eye.controller_view = bridge->engine->createView();" in source
    assert "eye.controller_guide_view = bridge->engine->createView();" in source
    assert "bridge->engine->destroy(bridge->renderer);" in source
    assert "bridge->engine->destroy(eye.renderer);" not in source
    assert "bridge->engine->destroy(eye.controller_view);" in source
    assert "bridge->engine->destroy(eye.controller_guide_view);" in source
    assert "filament::View* laser_view = nullptr;" not in source
    assert "display_view" not in source
    assert "eye.laser_view = bridge->engine->createView();" not in source
    assert "eye.view->setVisibleLayers(0xff, 0x03);" in source
    assert "eye.view->setChannelDepthClearEnabled(2, true);" in source
    assert "eye.foreground_view->setChannelDepthClearEnabled(2, true);" in source
    assert "eye.controller_view->setChannelDepthClearEnabled(2, true);" in source
    assert "eye.controller_guide_view->setChannelDepthClearEnabled(2, true);" in source
    assert "eye.foreground_view->setChannelDepthClearEnabled(0, true);" not in source
    assert "kScreenLayerBase + eye_index" in source
    assert "eye.controller_view->setVisibleLayers(0xff, 0x01);" in source
    assert "eye.controller_guide_view->setVisibleLayers(0xff, 0x04);" in source
    assert "Renderer::ClearOptions clear_options;" in source
    assert "clear_options.clear = true;" in source
    assert "eye.renderer->setClearOptions(clear_options);" in source
    preview_source = (bridge_dir / "preview_bridge.cpp").read_text(encoding="utf-8")
    assert "preview->renderer->setClearOptions(clear_options);" in preview_source
    assert "preview->view->setChannelDepthClearEnabled(2, true);" in preview_source
    assert "preview->indirect_light" in preview_source
    assert "filament_preview_set_ambient_light" in facade
    assert "bridge->renderer->render(bridge->view);" in source
    assert "bridge_controller_set_occlusion_materials" not in source
    assert "bridge_controller_create_occlusion_material" not in source
    assert "renderables.getMaterialInstanceAt" not in controller_source
    assert "renderables.setMaterialInstanceAt" not in controller_source
    assert "createInstancedAsset" not in source
    assert "createAsset(" in controller_source
    assert "FilamentInstance" not in source
    assert "bridge_set_renderable_layer" in source
    assert ".exposure(target->brightness.scene_exposure_ev)" in source
    assert "scene_factor" not in source
    assert "return configure_color_pipeline_impl(preview) ? 1 : 0;" in source
    assert "VK_FORMAT_R8G8B8A8_SRGB" in source
    assert "Virtual screen requires VK_FORMAT_R8G8B8A8_SRGB" in source
    assert "Display-referred screen content bypasses the HDR scene view." in source
    assert "add_screen_entities(bridge);" in source
    assert "bool screen_in_scene = false;" in source
    assert "The sampler is required by the material" in source
    assert "TextureSampler::MinFilter::LINEAR" in source
    assert "TextureSampler::MagFilter::LINEAR" in source
    assert "screen_texture_sampler.setAnisotropy(16.0f)" in source
    assert "screen_mip_experiment_enabled = true" in source
    assert 'parameter("screenTexelSize"' in source
    assert 'parameter("screenSourceSize"' in source
    assert 'parameter("screenOutputSize"' in source
    assert 'parameter("screenFilterScale"' in source
    assert 'parameter("screenSharpness"' in source
    assert 'parameter("screenExposureCompensation"' in source
    assert "materialParams.screenTexelSize" in source
    assert "materialParams_screenTexture" in source
    assert "materialParams.screenQualityPass" in source
    assert "materialParams.screenExposureCompensation" in source
    assert "materialParams_screenTexelSize" not in source
    assert "screen_lanczos2" in source
    assert "screen_easu" in source
    assert "quality_pass > 2.5 && quality_pass < 3.5" in source
    assert "screenQualityPass" in source
    assert "quality_pass > 0.5 && quality_pass < 1.5" in source
    assert "quality_pass > 1.5" in source
    assert "float rcas_limit = 0.25 - (1.0 / 16.0);" in source
    assert "vec3 lobe_rgb = max(-hit_min, hit_max);" in source
    assert "float sharpness_stops = 2.0 * (1.0 - sharpness);" in source
    assert "screen_mip_copy_material_instance->setParameter(" in source
    assert '"screenSharpness", bridge->screen_filter_sharpness' in source
    assert "screen_lanczos_textures" in source
    assert "screen_lanczos_render_targets" in source
    assert "kLegacyScreenLodBias = -0.35f" in source
    assert '"screenLodBias", kLegacyScreenLodBias' in source
    assert "materialParams.screenLodBias" in source
    assert "bridge->renderer->render(view);" in source
    assert "Match the legacy OpenGL default" in source
    assert "bridge->screen_filter_scale <= 1.0001f" in source
    assert "fwidth(uv)" not in source
    assert "RenderTarget::Builder" in source
    assert "generateMipmaps(*bridge->engine)" in source
    assert "Allocation does not mean LOD 0 contains the current source" in source
    assert "if (weight_sum <= 0.000001) return f;" in source
    assert "Recreate targets lazily with the new output extent" in source
    assert "screen_mip_copy_view" in source
    assert "bridge_screen_prepare_frame(bridge);" in (bridge_dir / "bridge_eye.cpp").read_text(encoding="utf-8")
    assert "screen_mip_generation_count[eye_index]" in source
    assert "LINEAR_MIPMAP_LINEAR" in source
    assert "screen_texture_sampler.setAnisotropy(16.0f)" in source
    assert "v=0 is the bottom edge" in source
    context_source = (bridge_dir / "bridge_context.cpp").read_text(encoding="utf-8")
    assert "eye.view->setAntiAliasing(filament::AntiAliasing::NONE);" in context_source
    assert "filament_bridge_set_screen_ready_semaphore" in facade
    assert "filament_bridge_set_screen_curved" in facade
    assert "kScreenSegments = 48" in source
    assert "bridge->screen_curved" in source
    assert "filament_bridge_set_passthrough_backdrop" in facade
    assert "skybox_entities" in source
    assert "float4{0.0f, 0.6f, 0.2f, 1.0f}" in source
    assert "screen_source_bind_count" in source
    assert "screen_mip_generation_count" in source
    assert "bridge_screen_get_sampling_stats" in source
    assert "pending_ready_semaphore" in source
    assert "screen_texture_cache" in source
    assert "bridge->engine->flushAndWait();" in source
    assert "bridge->engine->flush();" in source
    assert "filament_bridge_end_frame_deferred" in facade
    assert "filament_bridge_finish_frame_batch" in facade
    assert "filament_bridge_screen_eye_renderables_abi_available" in facade
    assert "screen_material_instances" in source
    assert "screen_mip_copy_material_instances" in source
    assert "screen_entities" in source
    assert "screen_mip_copy_entities" in source
    assert "screen_mip_copy_views" in source
    assert "renderables.setMaterialInstanceAt" not in (
        bridge_dir / "bridge_eye.cpp"
    ).read_text(encoding="utf-8")
    assert "if (bridge->active_eye == 0)" in (
        bridge_dir / "bridge_eye.cpp"
    ).read_text(encoding="utf-8")
    assert "bridge_eye_finish_frame_batch" in source
    assert "diagnostic_frame_count < 8" in source
    assert "bridge->multiview_active || bridge->active_eye == 1" in source
    assert "filament_bridge_multiview_abi_available() { return 2; }" in facade
    assert "[FilamentBridge] acquired eye=" in source
    assert "filament_bridge_set_controller_visible" in facade
    assert "renderables.setLayerMask" in source
    assert "filament_bridge_set_controller_laser" in facade
    assert "filament_bridge_set_controller_guide_texture" in facade
    assert "filament_bridge_set_controller_guide" in facade
    assert "D2S Controller Guide" in source
    assert "guide.rgb *= guide.a;" in source
    assert "Texture::InternalFormat::SRGB8_A8" in source
    assert ".blending(filament::BlendingMode::TRANSPARENT)" in source
    assert ".depthWrite(false)" in source
    assert ".depthCulling(false)" in source
    assert "bridge_set_renderable_layer(bridge, bridge->controller_guide_entity, 2" in source
    assert "D2S Controller Laser" in source
    assert 'parameter("laser_time"' in source
    assert "materialParams.laser_time * 0.4" in source
    assert "fract(uv.y + materialParams.laser_time * 0.4)" in source
    assert ".blending(filament::BlendingMode::OPAQUE)" in source
    assert ".depthWrite(true)" in source
    assert "materialParams_laser_time" not in source
    assert 'parameter("time"' not in source
    assert "float3(0.0, 0.4, 1.0)" in source
    assert "float3(1.0, 0.0, 0.0)" in source
    assert "std::array<PreviewScreenVertex, 8> laser_vertices" in source
    assert "std::array<uint16_t, 12> laser_indices" in source
    assert "controller_quaternion_slerp" in source
    assert "controller.button_values[5]" in source
    assert "controller loaded hand=%u animations=%zu" in source
    assert "filament_bridge_set_glow_source" in facade
    assert "filament_bridge_set_glow_state" in facade
    assert "D2S Legacy Screen Glow" in source
    assert "D2S Legacy Frosted Glow" in source
    assert "D2S Legacy Surround Glow" in source
    assert "kGlowShellSegments = 96" in source
    assert "kGlowShellRadialSegments = 48" in source
    assert "for (uint32_t side = 0; side < 4; ++side)" in source
    assert "Four independent edge-to-rim strips" in source
    assert "spherical_interpolate" in source
    assert "{radial_t, 0.0f}" in source
    assert "inverse_distance_squared" in source
    assert "direction * intersection_distance" in source
    assert "const auto source_position = screen_surface" in source
    assert "const float surface_distance" in source
    assert "surface_position = radial == 0" in source
    assert "? source_position" in source
    assert "vec2 grid = vec2(8.0, 6.0);" in source
    assert "sampleRegionCell" in source
    assert "sampleRegionAverage" in source
    assert "blend = blend * blend" in source
    assert "float radialDistance = clamp(getUV1().x" in source
    assert "float seamFeather = mix(" not in source
    assert "float edgeField = exp2(-5.0 * radialDistance)" in source
    assert "vec3 shellColor = sampleRegionAverage(sourceUv);" in source
    assert "const float phi" not in source
    assert "screen_relative_uv" not in source
    shell_buffer = source.split(
        "bridge->glow_shell_vertex_buffer =", 1
    )[1].split("bridge->glow_shell_index_buffer =", 1)[0]
    assert "filament::VertexAttribute::UV1" in shell_buffer
    assert "narrow inward pixel band" in source
    assert '.parameter("glowColor"' not in source
    assert 'setParameter(\n                "glowColor"' not in source
    assert "material.baseColor = vec4(shellColor * glow, 1.0);" in source
    shell_material = source.split(
        'name("D2S Legacy Surround Glow")', 1
    )[1].split(".build(bridge->engine->getJobSystem())", 1)[0]
    assert ".blending(filament::BlendingMode::ADD)" in shell_material
    assert "bridge->glow_mode == 5" in source
    assert "bridge->glow_index_buffer, kMaxGlowIndices, 4" in source
    assert "bridge->glow_index_buffer, kMaxGlowIndices, 5" in source
    assert "bridge->frost_index_buffer, kMaxFrostIndices, 5" in source
    assert "bridge->foreground_scene->remove(bridge->glow_shell_entity);" in source
    assert "bridge->scene->addEntity(bridge->glow_shell_entity);" in source
    screen_module = (bridge_dir / "bridge_screen.cpp").read_text(encoding="utf-8")
    assert "bridge_glow_update_geometry(bridge);" in screen_module
    assert "Legacy surround is a screen-background effect" in source
    assert "Texture::InternalFormat::SRGB8_A8" in source
    assert "generateMipmaps(*bridge->engine)" in source
    assert "kFlatFrostDepthSteps = 8" in source
    assert "kFlatFrostEdgeSteps = 8" in source
    assert "four independent walls" in source
    assert "material.baseColor = vec4(color * glow, 1.0);" in source
    assert "material.baseColor = vec4(color * alpha, 1.0);" in source
    assert "const bool default_environment = bridge->asset == nullptr;" in source
    assert "enabled && default_environment &&" in source
    glow_source = (bridge_dir / "bridge_glow.cpp").read_text(encoding="utf-8")
    assert "createExternalImageFromVkImage" in glow_source
    assert "glow_cpu_source_texture" in glow_source
    assert "Usage::GEN_MIPMAPPABLE" in glow_source
    assert ".usage(glow_texture_usage)" in glow_source
    assert "glow_texture_cache" in glow_source
    assert "glow_source_external" in glow_source
    assert 'parameter("externalSource"' in glow_source
    assert "materialParams.externalSource > 0.5" in glow_source
    assert "filament_bridge_set_glow_image" in facade
    assert "kControllerValues" in source
    assert "bridge_controller_find_instance_entity" not in source
    assert "if (!bridge || !controller.asset || value_entity.isNull())" in source
    assert "controller.asset->getFirstEntityByName" in source
    assert "bridge->asset->getFirstEntityByName" not in (
        bridge_dir / "bridge_controller.cpp"
    ).read_text(encoding="utf-8")
    assert "if (controller.animations.empty())" in source
    assert "Controller GLB exposes no _value/_min/_max animation triplets" in source
    assert "renderables.setLightChannel(instance, 0, false);" in source
    assert "const uint8_t layer_mask = occlusion_instance ? 0x02 : 0x01;" not in source
    assert "? (instance_index == 1 ? 0x02 : 0x01)" not in source
    assert "renderables.setLayerMask(instance, 0xff, 0x03);" not in source
    assert "bridge_set_renderable_layer(bridge, entity, 0, true);" in source
    assert "ToneMapping::LINEAR" in source
    assert "ToneMapping::ACES_LEGACY" not in source
    assert "LightManager::Type::POINT" in source
    assert "kLegacyControllerCandelaScale = 10000.0f" in source
    assert "kControllerBaseLightWeight = 0.20f" in source
    assert "intensity * kLegacyControllerCandelaScale *" in source
    assert "0.55f * intensity * kLegacyControllerCandelaScale *" in source
    assert "eye_y + 0.05f" in source
    assert "eye_y + 0.45f" in source
    assert "eye_z - 0.18f" in source
    assert '"specularColorFactor"' not in source
    assert '"roughnessFactor", 0.4f' not in source
    assert "filament::IndirectLight::Builder()" in source
    assert ".irradiance(1, irradiance)" in source
    assert ".intensity(kLegacyAmbientLux)" in source
    assert "filament_bridge_set_ambient_light" in facade
    assert "filament_bridge_set_controller_ambient_light" in facade
    assert "foreground_scene" in source
    assert "foreground_view" in source
    assert "controller_view" in source
    assert "controller_guide_view" in source
    assert "eye.foreground_view->setColorGrading(nullptr);" in source
    assert "eye.foreground_view->setColorGrading(eye.color_grading);" in source
    assert "eye.controller_view->setColorGrading(eye.color_grading);" in source
    assert "eye.controller_guide_view->setColorGrading(eye.color_grading);" in source
    assert (
        "eye.foreground_view->setBlendMode(filament::View::BlendMode::TRANSLUCENT);"
        in source
    )
    assert "eye.foreground_view->setPostProcessingEnabled(true);" in source
    assert "eye.controller_view->setPostProcessingEnabled(true);" in source
    assert "eye.controller_guide_view->setPostProcessingEnabled(true);" in source
    assert "filament_bridge_set_screen_light" in facade
    assert "filament_bridge_set_screen_sampling" in facade
    assert "filament_bridge_set_screen_upscale" in facade
    assert "filament_bridge_set_screen_sampling_mode" in facade
    assert "filament_bridge_get_screen_sampling_stats" in facade
    assert "filament_bridge_set_fixed_screen_image" in facade
    assert "bridge_screen_set_sampling_mode" in source
    assert "LightManager::Type::FOCUSED_SPOT" in source
    assert "bridge->screen_light_direction = -forward;" in source
    assert "std::sqrt(width * width + height * height)" in source
    assert "lights.setColor(instance" in source
    assert "lights.setIntensityCandela(" in source
    assert ".lightChannel(0, false)" in source
    assert ".lightChannel(1, true)" in source


def test_legacy_glow_material_shaders_compile_with_pinned_filament(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    candidates = (
        root / "native/filament/sdk/windows/v1.74.0/matc.exe",
        root / "native/filament/sdk/linux/v1.74.0/filament/bin/matc",
        root / "native/filament/sdk/macos/v1.74.0/filament/bin/matc",
    )
    matc = next((path for path in candidates if path.is_file()), None)
    if matc is None:
        pytest.skip("pinned Filament matc is unavailable")
    source = (root / "native/filament/bridge/bridge_glow.cpp").read_text(
        encoding="utf-8"
    )
    shader_sources = {
        "glow": (
            re.search(
                r'const char\* glow_shader = R"FILAMENT\((.*?)\)FILAMENT";',
                source,
                re.DOTALL,
            ),
            "uv0",
            """
                { type : sampler2d, name : glowTexture },
                { type : float, name : glowIntensity },
                { type : float, name : externalSource },
                { type : float2, name : screenHalf },
                { type : float, name : glowInvRange },
                { type : float, name : glowInner },
                { type : float, name : innerOnly }
            """,
        ),
        "frost": (
            re.search(
                r'const char\* frost_shader = R"FILAMENT\((.*?)\)FILAMENT";',
                source,
                re.DOTALL,
            ),
            "uv0, uv1",
            """
                { type : sampler2d, name : glowTexture },
                { type : float, name : glowIntensity },
                { type : float, name : externalSource },
                { type : float, name : effectMode },
                { type : float, name : effectAlpha },
                { type : float, name : effectThreshold },
                { type : float, name : effectLod },
                { type : float, name : effectBlend },
                { type : float, name : effectThickness },
                { type : float, name : effectDiffuse },
                { type : float, name : effectInset },
                { type : float, name : effectTime }
            """,
        ),
        "surround": (
            re.search(
                r'const char\* glow_shell_shader = R"FILAMENT\((.*?)\)FILAMENT";',
                source,
                re.DOTALL,
            ),
            "uv0, uv1",
            """
                { type : sampler2d, name : glowTexture },
                { type : float, name : glowIntensity },
                { type : float, name : externalSource }
            """,
        ),
    }
    for name, (match, requires, parameters) in shader_sources.items():
        assert match is not None
        material_path = tmp_path / f"{name}.mat"
        material_path.write_text(
            f"""
material {{
    name : D2S_{name}_validation,
    shadingModel : unlit,
    blending : transparent,
    doubleSided : true,
    depthWrite : false,
    depthCulling : false,
    requires : [ {requires} ],
    parameters : [ {parameters} ]
}}
fragment {{
{match.group(1)}
}}
""",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                str(matc), "--api", "all", "-o",
                str(tmp_path / f"{name}.filamat"), str(material_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_screen_light_is_independent_from_environment_hdr_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/xr_viewer/core_openxr_vulkan.py").read_text(
        encoding="utf-8"
    )
    method = source.split("    def _update_filament_screen_light(", 1)[1].split(
        "\n    def ", 1
    )[0]

    assert "screen_light_linear_rgb" in method
    assert "color = sampled" in method
    assert "sampled * 0.82 + neutral * 0.18" not in method
    assert "_controller_hdr_lighting" not in method
    assert "screen_light=always" in source
    assert "hdr_ibl_pending_profile_fallback" in source

    material_source = (
        root / "native/filament/bridge/bridge_material.cpp"
    ).read_text(encoding="utf-8")
    screen_source = (
        root / "native/filament/bridge/bridge_screen.cpp"
    ).read_text(encoding="utf-8")
    assert "kControllerBaseLightWeight = 0.20f" in material_source
    assert "kControllerScreenLightWeight = 0.80f" in screen_source


def test_artemis_controller_lighting_matches_legacy_head_light() -> None:
    root = Path(__file__).resolve().parents[1]
    profile_path = root / "src/xr_viewer/environments/3D_Artemis/profile.json"
    if not profile_path.is_file():
        profile_path = root / "src/xr_viewer/environments/Artemis/profile.json"
    profile = json.loads(
        profile_path.read_text(encoding="utf-8")
    )
    assert profile["env_head_light_color"] == [0.45, 0.45, 0.48]
    assert profile["env_ambient_color"] == [0.06, 0.05, 0.05]
    assert profile["preview_exposure"] == 0.0
    config = (root / "src/xr_viewer/core_openxr_vulkan.py").read_text(
        encoding="utf-8"
    )
    assert "filament_fill_light_intensity: float = 1.0" in config
    assert 'profile.get("env_head_light_color"' in config
    assert '"env_ambient_color", self._filament_ambient_light_color' in config
    assert "bridge.set_ambient_light(self._controller_ambient_light_color())" in config


@pytest.mark.parametrize("brand", ("HP", "INDEX", "PICO", "QUEST", "VIVE", "YVR"))
@pytest.mark.parametrize("hand", ("left", "right"))
def test_packaged_controller_glb_animation_triplets_have_native_fallbacks(
    brand: str, hand: str
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/xr_viewer/controllers"
        / brand
        / f"{hand}.glb"
    )
    payload = path.read_bytes()
    assert payload[:4] == b"glTF"
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(
        payload[20 : 20 + json_length].decode("utf-8").rstrip("\x00 ")
    )
    names = {str(node.get("name") or "") for node in document["nodes"]}
    value_names = {
        name
        for name in names
        if name.endswith("_value")
        and all(
            name.removesuffix("_value") + suffix in names
            for suffix in ("_min", "_max")
        )
    }

    assert value_names
    assert any("trigger_pressed_value" in name for name in value_names)
    assert any("squeeze_pressed_value" in name for name in value_names)
    assert any("thumbstick_pressed_value" in name for name in value_names)
    bridge_source = (
        Path(__file__).resolve().parents[1]
        / "native/filament/bridge/bridge_controller.cpp"
    ).read_text(encoding="utf-8")
    assert all(f'"{value_name}"' in bridge_source for value_name in value_names)


def test_native_controller_animation_preserves_touch_semantics_without_abi_growth() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/filament/bridge/bridge_controller.cpp").read_text(
        encoding="utf-8"
    )
    internal = (root / "native/filament/bridge/bridge_internal.h").read_text(
        encoding="utf-8"
    )
    public_header = (root / "native/filament/bridge/filament_bridge.h").read_text(
        encoding="utf-8"
    )

    assert '"joystick_x_touched"' in source
    assert '"joystick_y_touched"' in source
    assert '"joystick_touched"' in source
    assert "controller.button_values[6] * controller.joystick_x" in source
    assert "controller.button_values[6] * controller.joystick_y" in source
    assert "std::array<float, 7> button_values{};" in internal
    assert "uint32_t button_mask" in public_header


def test_native_foreground_uses_one_view_for_multiview_stereo() -> None:
    root = Path(__file__).resolve().parents[1]
    screen_source = (root / "native/filament/bridge/bridge_screen.cpp").read_text(
        encoding="utf-8"
    )
    glow_source = (root / "native/filament/bridge/bridge_glow.cpp").read_text(
        encoding="utf-8"
    )
    controller_source = (root / "native/filament/bridge/bridge_controller.cpp").read_text(
        encoding="utf-8"
    )
    laser_source = (root / "native/filament/bridge/bridge_laser.cpp").read_text(
        encoding="utf-8"
    )
    eye_source = (root / "native/filament/bridge/bridge_eye.cpp").read_text(
        encoding="utf-8"
    )

    assert ".priority(2).culling(false)" in screen_source
    assert "surround shell is the background effect" in screen_source
    assert ".blending(filament::BlendingMode::OPAQUE)" in screen_source
    assert ".depthWrite(false)" in screen_source
    assert ".depthCulling(true)" in screen_source
    assert ".blending(filament::BlendingMode::TRANSPARENT)" in glow_source
    assert "renderables.setPriority(instance, 6);" in controller_source
    assert ".priority(7)" in laser_source
    assert ".depthWrite(true)" in laser_source
    room = eye_source.index("bridge->renderer->render(bridge->view);")
    glow = eye_source.index("bridge->renderer->render(eye.foreground_view);")
    controllers = eye_source.index("bridge->renderer->render(eye.controller_view);")
    guide = eye_source.index(
        "bridge->renderer->render(eye.controller_guide_view);"
    )
    assert room < glow < controllers < guide
    assert "0x01u | 0x02u | 0x04u | (1u << kScreenLayerBase)" in eye_source
    assert "if (!bridge->multiview_active) {" in eye_source


def test_native_screen_has_opt_in_multiview_eye_diagnostic() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/filament/bridge/bridge_screen.cpp").read_text(
        encoding="utf-8"
    )

    assert "D2S_FILAMENT_EYE_DIAGNOSTIC" in source
    assert 'parameter("screenEyeDiagnostic"' in source
    assert "getEyeIndex() == 0" in source
    assert "left=red right=green" in source
