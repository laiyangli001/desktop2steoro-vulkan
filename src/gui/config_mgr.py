"""GUI Config Mixin — config read/write, stereo preset data, hot-param save."""
import os
import asyncio
from utils import OS_NAME, ALL_MODELS, DEFAULT_PORT, read_yaml
from utils.xr_headset_presets import display_to_xr_headset, xr_headset_to_display
from .config import (
    DEFAULTS, DEFAULT_FAMILIES, DEFAULT_MODEL_LIST, FAMILY_TO_SIZES,
    environment_display_label, parse_model_name, save_yaml,
)
from .paths import BASE_DIR, DIAG_LOG
from .localization import UI_MESSAGES
from .capture_sources import get_monitor_index_for_point, get_primary_monitor_index


class GUIConfigMixin:
    """Mixin providing config management for Desktop2StereoGUI."""

    # ── config apply ──

    def apply_config(self, cfg, keep_optional=True):
        self._config = cfg.copy()
        self._config.pop("Debug Mode", None)
        current_primary = get_primary_monitor_index()
        mon_idx = cfg.get("Monitor Index", DEFAULTS["Monitor Index"])
        label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == mon_idx), None)
        if label:
            self.monitor_dd.value = label
        elif self.monitor_label_to_index:
            primary_label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == current_primary), None)
            self.monitor_dd.value = primary_label or next(iter(self.monitor_label_to_index))
        self.selected_window_name = cfg.get("Window Title", "")
        self.selected_window_handle = None
        self.selected_window_rect = None
        if keep_optional and self.capture_mode_key == "Window":
            self.refresh_window_list()
            dev_idx = cfg.get("Computing Device", DEFAULTS["Computing Device"])
            dev_label = next((lbl for lbl, i in self.device_label_to_index.items() if i == dev_idx), None)
            if dev_label:
                self.device_dd.value = dev_label
        model_list = DEFAULT_MODEL_LIST
        selected_model = cfg.get("Depth Model", DEFAULTS["Depth Model"])
        if selected_model not in model_list:
            selected_model = model_list[0] if model_list else DEFAULTS["Depth Model"]
        family, size = parse_model_name(selected_model)
        if family not in DEFAULT_FAMILIES:
            family = DEFAULT_FAMILIES[0] if DEFAULT_FAMILIES else ""
        self.depth_model_dd.options = [f for f in DEFAULT_FAMILIES]
        self.depth_model_dd.value = family
        avail_sizes = FAMILY_TO_SIZES.get(family, [])
        self.model_size_dd.options = [s for s in avail_sizes]
        if size in avail_sizes:
            self.model_size_dd.value = size
        elif avail_sizes:
            self.model_size_dd.value = avail_sizes[0]
        else:
            self.model_size_dd.value = ""
        self.depth_res_dd.value = str(cfg.get("Depth Resolution", DEFAULTS["Depth Resolution"]))
        self.update_depth_resolution_options(self.current_model_name)
        depth_strength = self._clamp_depth_strength(
            cfg.get("Depth Strength", DEFAULTS["Depth Strength"]))
        self.depth_strength_dd.value = f"{depth_strength:.2f}"
        self.depth_quick_dd.value = self._depth_quick_to_display(
            cfg.get("Depth Quick", self._depth_quick_from_strength(depth_strength)))
        self.display_mode_dd.value = cfg.get("Display Mode", DEFAULTS["Display Mode"])
        self.xr_headset_dd.value = xr_headset_to_display(
            cfg.get("XR Headset Model", DEFAULTS["XR Headset Model"]), self.locale)
        self.xr_preview_cb.value = cfg.get("XR Preview Window", DEFAULTS["XR Preview Window"])
        self.local_vsync_cb.value = cfg.get("VSync", DEFAULTS["VSync"])
        self.advanced_device_cb.value = False
        self.upscaler_dd.options = self._upscaler_display_options()
        self.upscaler_dd.value = self._upscaler_to_display("Off")
        self.upscaler_sharpness_dd.value = "0.00"
        target_fps = self._parse_int(cfg.get("Target FPS", DEFAULTS["Target FPS"]), DEFAULTS["Target FPS"])
        self.target_fps_dd.value = self._target_fps_to_display(target_fps)
        self.render_policy_dd.value = self._render_policy_to_display(
            cfg.get("Render Size Policy", DEFAULTS["Render Size Policy"]))
        self.render_scale_dd.value = self._render_scale_to_display(
            cfg.get("Render Scale", DEFAULTS["Render Scale"]))
        fixed_width = self._parse_int(cfg.get("Render Fixed Width", DEFAULTS["Render Fixed Width"]), DEFAULTS["Render Fixed Width"])
        fixed_height = self._parse_int(cfg.get("Render Fixed Height", DEFAULTS["Render Fixed Height"]), DEFAULTS["Render Fixed Height"])
        self.render_fixed_dd.value = self._fixed_size_to_display(fixed_width, fixed_height)
        self.render_max_pixels_dd.value = str(self._parse_int(
            cfg.get("Render Max Pixels", DEFAULTS["Render Max Pixels"]), DEFAULTS["Render Max Pixels"]))
        self.render_min_dimension_dd.value = str(self._parse_int(
            cfg.get("Render Min Dimension", DEFAULTS["Render Min Dimension"]), DEFAULTS["Render Min Dimension"]))
        self.render_align_dd.value = str(self._parse_int(cfg.get("Render Align", DEFAULTS["Render Align"]), DEFAULTS["Render Align"]))
        self.antialiasing_dd.value = str(cfg.get("Anti-aliasing", DEFAULTS["Anti-aliasing"]))
        self.depth_pop_dd.value = str(self._clamp_depth_pop(
            cfg.get("Depth Pop", DEFAULTS["Depth Pop"])))
        self.foreground_pop_dd.value = f'{self._parse_float(cfg.get("Foreground Pop", DEFAULTS["Foreground Pop"]), DEFAULTS["Foreground Pop"]):.2f}'
        self.midground_pop_dd.value = f'{self._parse_float(cfg.get("Midground Pop", DEFAULTS["Midground Pop"]), DEFAULTS["Midground Pop"]):.2f}'
        self.background_pop_dd.value = f'{self._parse_float(cfg.get("Background Pop", DEFAULTS["Background Pop"]), DEFAULTS["Background Pop"]):.2f}'
        self.depth_separation_dd.value = self._depth_separation_to_display(
            cfg.get("Depth Separation Preset", DEFAULTS["Depth Separation Preset"]))
        self.convergence_dd.value = str(cfg.get("Convergence", DEFAULTS["Convergence"]))
        self.dynamic_convergence_strength_dd.value = f'{self._parse_float(cfg.get("Dynamic Convergence Strength", DEFAULTS["Dynamic Convergence Strength"]), DEFAULTS["Dynamic Convergence Strength"]):.2f}'
        stereo_preset = self._display_to_preset(cfg.get("Stereo Preset", DEFAULTS["Stereo Preset"]))
        self.stereo_preset_dd.value = self._preset_to_display(stereo_preset)
        self.stereo_quality_dd.value = self._stereo_quality_to_display(
            self._stereo_quality_for_preset(stereo_preset))
        self.parallax_budget_dd.value = self._parallax_budget_to_display(
            cfg.get("Parallax Budget Preset", cfg.get("Parallax Preset", "standard")))
        self.temporal_strength_dd.value = f'{self._parse_float(cfg.get("Temporal Strength", DEFAULTS["Temporal Strength"]), DEFAULTS["Temporal Strength"]):.2f}'
        self.edge_dilation_dd.value = str(cfg.get("Edge Dilation", DEFAULTS["Edge Dilation"]))
        self.mask_feather_dd.value = str(cfg.get("Mask Feather Radius", DEFAULTS["Mask Feather Radius"]))
        self.hole_fill_mode_dd.value = self._hole_fill_mode_to_display(cfg.get("Hole Fill Mode", DEFAULTS["Hole Fill Mode"]))
        self.edge_threshold_dd.value = f'{self._parse_float(cfg.get("Edge Threshold", DEFAULTS["Edge Threshold"]), DEFAULTS["Edge Threshold"]):.2f}'
        self.cross_eyed_cb.value = cfg.get("Cross Eyed", DEFAULTS["Cross Eyed"])
        self.anaglyph_dd.value = cfg.get("Anaglyph Method", DEFAULTS["Anaglyph Method"])
        self.advanced_stereo_cb.value = False
        self._sync_advanced_stereo_visibility()
        self._sync_device_advanced_visibility(cfg.get("Run Mode", DEFAULTS.get("Run Mode", "Local Viewer")))
        self.fp16_cb.value = bool(cfg.get("FP16", DEFAULTS["FP16"]))
        self.showfps_cb.value = cfg.get("Show FPS", DEFAULTS["Show FPS"])
        self.fill_16_9_cb.value = cfg.get("Fill 16:9", DEFAULTS["Fill 16:9"])
        self.fix_aspect_cb.value = cfg.get("Fix Viewer Aspect", DEFAULTS["Fix Viewer Aspect"])
        self.lossless_cb.value = cfg.get("Lossless Scaling Support", DEFAULTS["Lossless Scaling Support"])
        if keep_optional:
            self.locale = cfg.get("Language", DEFAULTS["Language"])
            self.lang_dd.value = "English" if self.locale == "EN" else "简体中文"

        saved_ctrl = cfg.get("Controller Model", DEFAULTS.get("Controller Model", "PICO"))
        self.ctrl_model_dd.value = saved_ctrl if saved_ctrl in self.ctrl_model_dd.options else "PICO"
        saved_env = cfg.get("Environment Model", DEFAULTS.get("Environment Model", "Default"))
        if str(saved_env).strip().lower() == "none":
            saved_env = "Default"
        self.env_key = saved_env if saved_env in self.env_model_keys else (self.env_model_keys[0] if self.env_model_keys else "Default")
        self.env_model_dd.value = environment_display_label(self.env_key, self.locale, self.env_model_display_names)
        self.torch_compile_cb.value = cfg.get("torch.compile")
        if self.torch_compile_cb.value is None:
            self.torch_compile_cb.value = False
        trt_val = cfg.get("TensorRT")
        if trt_val is not None:
            self.tensorrt_cb.value = trt_val
        self.recompile_trt_cb.value = cfg.get("Recompile TensorRT", DEFAULTS["Recompile TensorRT"])
        mgx_val = cfg.get("MIGraphX")
        if mgx_val is not None:
            self.migraphx_cb.value = mgx_val
        self.recompile_migraphx_cb.value = cfg.get("Recompile MIGraphX", DEFAULTS["Recompile MIGraphX"])
        cml_val = cfg.get("CoreML")
        if cml_val is not None:
            self.coreml_cb.value = cml_val
        self.recompile_coreml_cb.value = cfg.get("Recompile CoreML", DEFAULTS["Recompile CoreML"])
        ov_val = cfg.get("OpenVINO")
        if ov_val is not None:
            self.openvino_cb.value = ov_val
        self.recompile_openvino_cb.value = cfg.get("Recompile OpenVINO", DEFAULTS["Recompile OpenVINO"])
        self.recompile_trt_cb.visible = self.tensorrt_cb.value and self.tensorrt_cb.visible
        self.recompile_migraphx_cb.visible = self.migraphx_cb.value and self.migraphx_cb.visible
        self.recompile_coreml_cb.visible = self.coreml_cb.value and self.coreml_cb.visible
        self.recompile_openvino_cb.visible = self.openvino_cb.value and self.openvino_cb.visible
        ct = cfg.get("Capture Tool", DEFAULTS["Capture Tool"])
        self.capture_tool_dd.value = ct if ct in self.capture_tool_dd.options else (self.capture_tool_dd.options[0] if self.capture_tool_dd.options else '')
        if keep_optional:
            run_mode = cfg.get("Run Mode", DEFAULTS.get("Run Mode", "Local Viewer"))
            if run_mode == "3D Monitor" and OS_NAME != "Windows":
                run_mode = "Local Viewer"
            if run_mode == "OpenXR Link" and OS_NAME == "Darwin":
                run_mode = "Local Viewer"
            self.run_mode_key = run_mode
        self.stream_protocol_key = cfg.get("Stream Protocol", DEFAULTS.get("Stream Protocol", "RTMP"))
        self.stream_proto_dd.value = self.stream_protocol_key
        self.stream_port_tf.value = str(cfg.get("Streamer Port", DEFAULTS.get("Streamer Port", DEFAULT_PORT)))
        self.stream_quality_dd.value = str(cfg.get("Stream Quality", DEFAULTS["Stream Quality"]))
        self.stream_key_tf.value = cfg.get("Stream Key", DEFAULTS["Stream Key"])
        self.audio_dd.value = cfg.get("Stereo Mix", "")
        self.crf_tf.value = str(cfg.get("CRF", DEFAULTS["CRF"]))
        self.audio_delay_tf.value = str(cfg.get("Audio Delay", DEFAULTS["Audio Delay"]))
        self.capture_mode_key = cfg.get("Capture Mode", DEFAULTS["Capture Mode"])
        cm_t = UI_MESSAGES[self.locale]
        self.capture_mode_dd.value = cm_t["Monitor"] if self.capture_mode_key == "Monitor" else cm_t["Window"]
        self._sync_capture_mode_visibility()
        self._apply_stereo_output(cfg)
        self.update_tensorrt_visibility_based_on_model(selected_model)
        self.update_migraphx_visibility_based_on_model(selected_model)
        self.update_coreml_visibility_based_on_model(selected_model)
        self.update_openvino_visibility_based_on_model(selected_model)
        self.update_ui_texts()
        self._sync_visibility()
        self.update_stream_url()
        self.on_device_change(None)
        self.on_capture_tool_change(None)

    # ── config collect ──

    def _collect_config(self):
        if self.capture_mode_key == "Window":
            window_rect = getattr(self, 'selected_window_rect', None)
            if window_rect:
                center_x = window_rect[0] + window_rect[2] // 2
                center_y = window_rect[1] + window_rect[3] // 2
                monitor_idx = get_monitor_index_for_point(center_x, center_y)
            else:
                monitor_idx = get_primary_monitor_index()
        else:
            monitor_idx = self.monitor_label_to_index.get(self.monitor_dd.value, DEFAULTS["Monitor Index"])
        stereo_val = self.stereo_monitor_dd.value
        if stereo_val == "Viewer Window" or not stereo_val:
            stereo_idx = None
        else:
            stereo_idx = self.monitor_label_to_index.get(stereo_val, None)

        temporal_strength = self._parse_float(self.temporal_strength_dd.value, DEFAULTS["Temporal Strength"])
        scene_reset_threshold = self._parse_float(self.scene_reset_dd.value, DEFAULTS["Scene Reset Threshold"])
        depth_pop = self._clamp_depth_pop(self._parse_float(self.depth_pop_dd.value, DEFAULTS["Depth Pop"]))
        dynamic_convergence_strength = self._parse_float(self.dynamic_convergence_strength_dd.value, DEFAULTS["Dynamic Convergence Strength"])
        accelerator_values, recompile_values = self._platform_accelerator_values()
        fp16_value = False if "MPS" in (self.device_dd.value or "") else bool(self.fp16_cb.value)

        stereo_preset = self._display_to_preset(self.stereo_preset_dd.value)
        preset_values = self._stereo_preset_gui_values(stereo_preset) or {}
        stereo_quality = self._stereo_quality_for_preset(stereo_preset)
        parallax_budget = self._display_to_parallax_budget(self.parallax_budget_dd.value)
        render_fixed_width, render_fixed_height = self._parse_fixed_size(self.render_fixed_dd.value)
        self._config.pop("Debug Mode", None)

        self._config.update({
            "Capture Mode": self.capture_mode_key,
            "Monitor Index": monitor_idx,
            "Window Title": self.selected_window_name if self.capture_mode_key == "Window" else "",
            "Show FPS": self.showfps_cb.value,
            "Stereo Preset": stereo_preset,
            "Stereo Quality": stereo_quality,
            "Synthetic View": stereo_quality,
            "Parallax Budget Preset": parallax_budget,
            "Convergence": self._parse_float(self.convergence_dd.value, DEFAULTS["Convergence"]),
            "Dynamic Convergence": dynamic_convergence_strength > 0.0,
            "Dynamic Convergence Strength": dynamic_convergence_strength,
            "Display Mode": self.display_mode_dd.value,
            "Model List": ALL_MODELS,
            "Depth Model": self.current_model_name,
            "Depth Strength": self._clamp_depth_strength(self.depth_strength_dd.value),
            "Depth Quick": self._display_to_depth_quick(self.depth_quick_dd.value),
            "Anti-aliasing": self._parse_int(self.antialiasing_dd.value, DEFAULTS["Anti-aliasing"]),
            "Depth Antialias Strength": self._parse_float(self.antialiasing_dd.value, DEFAULTS["Depth Antialias Strength"]),
            "Temporal": temporal_strength > 0.0,
            "Temporal Strength": temporal_strength,
            "Auto Scene Reset": scene_reset_threshold > 0.0,
            "Scene Reset Threshold": scene_reset_threshold,
            "Edge Dilation": self._parse_int(self.edge_dilation_dd.value, DEFAULTS["Edge Dilation"]),
            "Mask Feather Radius": self._parse_int(self.mask_feather_dd.value, DEFAULTS["Mask Feather Radius"]),
            "Hole Fill Mode": self._display_to_hole_fill_mode(self.hole_fill_mode_dd.value),
            "Hole Fill Radius": int(preset_values.get("hole_fill_radius", DEFAULTS["Hole Fill Radius"])),
            "Hole Fill Strength": float(preset_values.get("hole_fill_strength", DEFAULTS["Hole Fill Strength"])),
            "Edge Threshold": self._parse_float(self.edge_threshold_dd.value, DEFAULTS["Edge Threshold"]),
            "Cross Eyed": self.cross_eyed_cb.value,
            "Anaglyph Method": self.anaglyph_dd.value,
            "Depth Pop": depth_pop,
            "Foreground Pop": self._parse_float(self.foreground_pop_dd.value, DEFAULTS["Foreground Pop"]),
            "Midground Pop": self._parse_float(self.midground_pop_dd.value, DEFAULTS["Midground Pop"]),
            "Background Pop": self._parse_float(self.background_pop_dd.value, DEFAULTS["Background Pop"]),
            "Depth Separation Preset": self._display_to_depth_separation(self.depth_separation_dd.value),
            "Depth Resolution": self._parse_int(self.depth_res_dd.value, DEFAULTS["Depth Resolution"]),
            "FP16": fp16_value,
            "Computing Device": self.device_label_to_index.get(self.device_dd.value, DEFAULTS["Computing Device"]),
            "Language": self.locale,
            "Run Mode": self.run_mode_key,
            "XR Headset Model": display_to_xr_headset(self.xr_headset_dd.value),
            "XR Preview Window": self.xr_preview_cb.value,
            "VSync": self.local_vsync_cb.value,
            "Target FPS": self._target_fps_from_display(self.target_fps_dd.value),
            "Processing Resolution": self._config.get("Processing Resolution", DEFAULTS["Processing Resolution"]),
            "Render Size Policy": "scaled",
            "Render Scale": self._display_to_render_scale(self.render_scale_dd.value),
            "Render Fixed Width": render_fixed_width,
            "Render Fixed Height": render_fixed_height,
            "Render Max Pixels": self._parse_int(self.render_max_pixels_dd.value, DEFAULTS["Render Max Pixels"]),
            "Render Min Dimension": self._parse_int(self.render_min_dimension_dd.value, DEFAULTS["Render Min Dimension"]),
            "Render Align": self._parse_int(self.render_align_dd.value, DEFAULTS["Render Align"]),
            "Upscaler": "Off",
            "Upscaler Sharpness": 0.0,
            "Stream Protocol": self.stream_proto_dd.value,
            "Streamer Port": self._parse_int(self.stream_port_tf.value, DEFAULTS["Streamer Port"]),
            "Stream Quality": self._parse_int(self.stream_quality_dd.value, DEFAULTS["Stream Quality"]),
            "torch.compile": self.torch_compile_cb.value,
            **accelerator_values,
            **recompile_values,
            "Capture Tool": self.capture_tool_dd.value,
            "Fill 16:9": self.fill_16_9_cb.value,
            "Fix Viewer Aspect": self.fix_aspect_cb.value,
            "Lossless Scaling Support": self.lossless_cb.value,
            "Stream Key": self.stream_key_tf.value,
            "Stereo Mix": self.audio_dd.value,
            "CRF": self._parse_int(self.crf_tf.value, DEFAULTS["CRF"]),
            "Audio Delay": self._parse_float(self.audio_delay_tf.value, DEFAULTS["Audio Delay"]),
            "Stereo Output": stereo_idx,
            "Controller Model": self.ctrl_model_dd.value,
            "Environment Model": self.env_key,
        })
        self.recompile_trt_cb.value = False
        self.recompile_migraphx_cb.value = False
        self.recompile_coreml_cb.value = False
        self.recompile_openvino_cb.value = False

    # ── stereo hot-param save ──

    def on_stereo_hot_param_change(self, e=None):
        self._schedule_stereo_hot_save()

    def _schedule_stereo_hot_save(self, delay=0.15):
        task = getattr(self, "_hot_save_task", None)
        if task and not task.done():
            task.cancel()
        try:
            self._hot_save_task = asyncio.create_task(self._save_stereo_hot_params_after_delay(delay))
        except RuntimeError:
            self._save_stereo_hot_params()

    async def _save_stereo_hot_params_after_delay(self, delay):
        try:
            await asyncio.sleep(delay)
            self._save_stereo_hot_params()
        except asyncio.CancelledError:
            return

    def _save_stereo_hot_params(self):
        path = os.path.join(BASE_DIR, "settings.yaml")
        cfg = self._config.copy()
        if os.path.exists(path):
            try:
                loaded = read_yaml(path)
                if loaded:
                    cfg.update(loaded)
            except Exception:
                pass
        cfg.pop("Debug Mode", None)
        temporal_strength = self._parse_float(self.temporal_strength_dd.value, DEFAULTS["Temporal Strength"])
        scene_reset_threshold = self._parse_float(self.scene_reset_dd.value, DEFAULTS["Scene Reset Threshold"])
        antialias_strength = self._parse_float(self.antialiasing_dd.value, DEFAULTS["Depth Antialias Strength"])
        depth_pop = self._clamp_depth_pop(self._parse_float(self.depth_pop_dd.value, DEFAULTS["Depth Pop"]))
        self.depth_pop_dd.value = f"{depth_pop:.1f}"
        dynamic_convergence_strength = self._parse_float(self.dynamic_convergence_strength_dd.value, DEFAULTS["Dynamic Convergence Strength"])
        stereo_preset = self._display_to_preset(self.stereo_preset_dd.value)
        preset_values = self._stereo_preset_gui_values(stereo_preset) or {}
        stereo_quality = self._stereo_quality_for_preset(stereo_preset)
        parallax_budget = self._display_to_parallax_budget(self.parallax_budget_dd.value)

        cfg.update({
            "Stereo Preset": stereo_preset,
            "Stereo Quality": stereo_quality,
            "Synthetic View": stereo_quality,
            "Parallax Budget Preset": parallax_budget,
            "Convergence": self._parse_float(self.convergence_dd.value, DEFAULTS["Convergence"]),
            "Dynamic Convergence": dynamic_convergence_strength > 0.0,
            "Dynamic Convergence Strength": dynamic_convergence_strength,
            "Depth Strength": self._clamp_depth_strength(self.depth_strength_dd.value),
            "Depth Quick": self._display_to_depth_quick(self.depth_quick_dd.value),
            "Temporal": temporal_strength > 0.0,
            "Temporal Strength": temporal_strength,
            "Auto Scene Reset": scene_reset_threshold > 0.0,
            "Scene Reset Threshold": scene_reset_threshold,
            "Depth Pop": depth_pop,
            "Foreground Pop": self._parse_float(self.foreground_pop_dd.value, DEFAULTS["Foreground Pop"]),
            "Midground Pop": self._parse_float(self.midground_pop_dd.value, DEFAULTS["Midground Pop"]),
            "Background Pop": self._parse_float(self.background_pop_dd.value, DEFAULTS["Background Pop"]),
            "Depth Separation Preset": self._display_to_depth_separation(self.depth_separation_dd.value),
            "Anti-aliasing": self._parse_int(self.antialiasing_dd.value, DEFAULTS["Anti-aliasing"]),
            "Depth Antialias Strength": antialias_strength,
            "Edge Dilation": self._parse_int(self.edge_dilation_dd.value, DEFAULTS["Edge Dilation"]),
            "Mask Feather Radius": self._parse_int(self.mask_feather_dd.value, DEFAULTS["Mask Feather Radius"]),
            "Hole Fill Mode": self._display_to_hole_fill_mode(self.hole_fill_mode_dd.value),
            "Hole Fill Radius": int(preset_values.get("hole_fill_radius", DEFAULTS["Hole Fill Radius"])),
            "Hole Fill Strength": float(preset_values.get("hole_fill_strength", DEFAULTS["Hole Fill Strength"])),
            "Edge Threshold": self._parse_float(self.edge_threshold_dd.value, DEFAULTS["Edge Threshold"]),
            "Anaglyph Method": self.anaglyph_dd.value,
            "Cross Eyed": bool(self.cross_eyed_cb.value),
        })
        ok, err = save_yaml(path, cfg)
        if ok:
            self._config.update(cfg)
            self.set_status(UI_MESSAGES[self.locale]["stereo_parameters_saved"], key="stereo_parameters_saved")
        else:
            self.set_status(UI_MESSAGES[self.locale]["failed_save_yaml"].format(err))

    # ── stereo preset values (static data) ──

    @staticmethod
    def _stereo_preset_gui_values(preset):
        presets = {
            "traditional_fastest": {
                "quality": "fast", "parallax_budget": "standard", "depth_strength": 0.25, "depth_quick": "Standard",
                "convergence": 0.0, "dynamic_convergence": False, "dynamic_convergence_strength": 0.0,
                "temporal_strength": 0.0, "scene_reset_threshold": 0.22,
                "depth_pop": 0.0, "depth_separation": "default", "foreground_pop": 1.0, "midground_pop": 1.0, "background_pop": 1.0, "antialiasing": 0, "depth_antialias_strength": 0.0,
                "edge_dilation": 0, "mask_feather_radius": 0, "hole_fill_mode": "balanced",
                "hole_fill_radius": 0, "hole_fill_strength": 0.0, "edge_threshold": 0.04,
                "cross_eyed": False,
            },
            "cinema": {
                "quality": "quality_4k", "parallax_budget": "standard", "depth_strength": 0.25, "depth_quick": "Standard",
                "convergence": 0.0, "dynamic_convergence": False, "dynamic_convergence_strength": 0.0,
                "temporal_strength": 0.25, "scene_reset_threshold": 0.22,
                "depth_pop": 0.0, "depth_separation": "standard", "foreground_pop": 1.15, "midground_pop": 1.05, "background_pop": 1.05, "antialiasing": 1, "depth_antialias_strength": 1.0,
                "edge_dilation": 1, "mask_feather_radius": 1, "hole_fill_mode": "balanced",
                "hole_fill_radius": 1, "hole_fill_strength": 0.60, "edge_threshold": 0.04,
                "cross_eyed": False,
            },
            "game_low_latency": {
                "quality": "fast_plus", "parallax_budget": "comfort", "depth_strength": 0.20, "depth_quick": "Soft",
                "convergence": 0.0, "dynamic_convergence": False, "dynamic_convergence_strength": 0.0,
                "temporal_strength": 0.0, "scene_reset_threshold": 0.18,
                "depth_pop": 0.0, "depth_separation": "weak", "foreground_pop": 1.15, "midground_pop": 1.05, "background_pop": 0.85, "antialiasing": 0, "depth_antialias_strength": 0.0,
                "edge_dilation": 1, "mask_feather_radius": 0, "hole_fill_mode": "soft_low_ghost",
                "hole_fill_radius": 1, "hole_fill_strength": 0.60, "edge_threshold": 0.04,
                "cross_eyed": False,
            },
            "still_image_hq": {
                "quality": "hq_4k", "parallax_budget": "strong", "depth_strength": 0.30, "depth_quick": "Enhanced",
                "convergence": 0.0, "dynamic_convergence": False, "dynamic_convergence_strength": 0.0,
                "temporal_strength": 0.0, "scene_reset_threshold": 0.00,
                "depth_pop": 0.0, "depth_separation": "strong", "foreground_pop": 1.25, "midground_pop": 1.10, "background_pop": 1.00, "antialiasing": 2, "depth_antialias_strength": 2.0,
                "edge_dilation": 3, "mask_feather_radius": 3, "hole_fill_mode": "quality",
                "hole_fill_radius": 3, "hole_fill_strength": 1.0, "edge_threshold": 0.04,
                "cross_eyed": False,
            },
            "debug_export": {
                "quality": "quality_4k", "parallax_budget": "standard", "depth_strength": 0.30, "depth_quick": "Enhanced",
                "convergence": 0.0, "dynamic_convergence": False, "dynamic_convergence_strength": 0.0,
                "temporal_strength": 0.0, "scene_reset_threshold": 0.22,
                "depth_pop": 0.0, "depth_separation": "default", "foreground_pop": 1.0, "midground_pop": 1.0, "background_pop": 1.0, "antialiasing": 0, "depth_antialias_strength": 0.0,
                "edge_dilation": 1, "mask_feather_radius": 0, "hole_fill_mode": "sharp_test",
                "hole_fill_radius": 1, "hole_fill_strength": 0.45, "edge_threshold": 0.04,
                "cross_eyed": False,
            },
        }
        return presets.get(preset)

    @staticmethod
    def _depth_separation_values(value):
        return {
            "default": (1.00, 1.00, 1.00),
            "standard": (1.15, 1.05, 1.05),
            "strong": (1.25, 1.10, 1.00),
            "weak": (1.15, 1.05, 0.85),
        }.get(value, (1.15, 1.05, 1.05))

    @classmethod
    def _stereo_quality_for_preset(cls, preset):
        values = cls._stereo_preset_gui_values(preset)
        if values:
            return values["quality"]
        return DEFAULTS["Stereo Quality"]

    @staticmethod
    def _depth_strength_for_quick(value):
        return {"Soft": 0.20, "Standard": 0.25, "Enhanced": 0.30}.get(value, 0.25)

    @staticmethod
    def _clamp_depth_strength(value):
        try:
            strength = float(value)
        except (TypeError, ValueError):
            strength = float(DEFAULTS["Depth Strength"])
        return max(0.0, min(0.5, strength))

    # ── stereo quality converters (delegate to localization module) ──

    def _stereo_quality_options(self):
        from .localization import stereo_quality_options
        return stereo_quality_options(self.locale)

    def _stereo_quality_to_display(self, value):
        from .localization import stereo_quality_to_display
        return stereo_quality_to_display(value, self.locale)

    def _parallax_budget_options(self):
        from .localization import parallax_budget_options
        return parallax_budget_options(self.locale)

    def _parallax_budget_to_display(self, value):
        from .localization import parallax_budget_to_display
        return parallax_budget_to_display(value, self.locale)

    def _hole_fill_mode_options(self):
        from .localization import hole_fill_mode_options
        return hole_fill_mode_options(self.locale)

    def _hole_fill_mode_to_display(self, value):
        from .localization import hole_fill_mode_to_display
        return hole_fill_mode_to_display(value, self.locale)

    def _depth_separation_options(self):
        from .localization import depth_separation_options
        return depth_separation_options(self.locale)

    def _depth_separation_to_display(self, value):
        from .localization import depth_separation_to_display
        return depth_separation_to_display(value, self.locale)

    @staticmethod
    def _display_to_depth_separation(value):
        from .localization import display_to_depth_separation
        return display_to_depth_separation(value)

    @staticmethod
    def _display_to_hole_fill_mode(value):
        from .localization import display_to_hole_fill_mode
        return display_to_hole_fill_mode(value)

    @staticmethod
    def _display_to_stereo_quality(value):
        from .localization import display_to_stereo_quality
        return display_to_stereo_quality(value)

    @staticmethod
    def _display_to_parallax_budget(value):
        from .localization import display_to_parallax_budget
        return display_to_parallax_budget(value)

    # ── static converters (no self.locale dependency) ──

    @staticmethod
    def _upscaler_from_display(value):
        return "Off"

    @staticmethod
    def _target_fps_from_display(value):
        value_l = str(value or "").strip().lower()
        if value_l in ("auto", "自动"):
            return 0
        try:
            return int(value_l)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _display_to_depth_quick(value):
        mapping = {
            "Soft": "Soft", "Standard": "Standard", "Enhanced": "Enhanced",
            "柔和": "Soft", "标准": "Standard", "增强": "Enhanced",
        }
        return mapping.get(value, str(value or "Standard"))

    @staticmethod
    def _depth_quick_from_strength(value):
        try:
            strength = float(value)
        except (TypeError, ValueError):
            return "Standard"
        if strength < 0.225:
            return "Soft"
        if strength > 0.275:
            return "Enhanced"
        return "Standard"

    @staticmethod
    def _display_to_preset(value):
        mapping = {
            "Traditional / Fastest": "traditional_fastest", "传统 / 速度快": "traditional_fastest",
            "Cinema": "cinema", "Cinema / Balance": "cinema", "影院": "cinema", "电影 / 偏均衡": "cinema",
            "Game / Low Latency": "game_low_latency", "游戏 / 低延迟": "game_low_latency",
            "Image  / High Quality": "still_image_hq", "图片 / 高质量": "still_image_hq",
            "Debug / Export": "cinema", "调试 / 导出": "cinema",
        }
        return mapping.get(value, str(value or "cinema").strip().lower())

    @staticmethod
    def _fixed_size_to_display(width, height):
        return f"{width}x{height}"

    @staticmethod
    def _parse_fixed_size(value):
        text = str(value or "").strip().lower().replace(" ", "")
        if "x" not in text:
            return DEFAULTS["Render Fixed Width"], DEFAULTS["Render Fixed Height"]
        width_text, height_text = text.split("x", 1)
        try:
            width = int(width_text)
            height = int(height_text)
        except (TypeError, ValueError):
            return DEFAULTS["Render Fixed Width"], DEFAULTS["Render Fixed Height"]
        if width <= 0 or height <= 0:
            return DEFAULTS["Render Fixed Width"], DEFAULTS["Render Fixed Height"]
        return width, height

    @staticmethod
    def _clamp_depth_pop(value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            value = float(DEFAULTS["Depth Pop"])
        return max(-0.9, min(5.0, value))

    @staticmethod
    def _parse_int(val, default):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _parse_float(val, default):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
