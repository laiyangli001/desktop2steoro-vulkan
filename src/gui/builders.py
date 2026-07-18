"""GUI Builder Mixin — UI construction, layout calculation, window sizing."""
import os
import flet as ft
from utils import OS_NAME, ALL_MODELS, DEFAULT_PORT
from utils.xr_headset_presets import xr_headset_options, xr_headset_to_display
from .config import (
    DEFAULTS, DEFAULT_FAMILIES, DEFAULT_MODEL_LIST,
    FAMILY_SIZE_TO_MODEL, FAMILY_TO_SIZES,
    environment_display_label, get_environment_model_options,
    load_environment_display_names, parse_model_name,
)
from .controls import FONT_SIZE, SCALE, CompactDropdown, CompactTextField, S, set_label_align_width
from .paths import BASE_DIR
from .localization import UI_MESSAGES
from .capture_sources import (
    PRIMARY_MONITOR_SUFFIX, get_capture_tool_options,
    get_primary_monitor_index, list_monitors, list_windows,
)
from .devices import DEVICES


class GUIBuilderMixin:
    """Mixin providing UI construction and layout sizing for Desktop2StereoGUI."""

    # ── width/height estimation ──

    def _ctrl_width(self, ctrl):
        """Get actual width of a control, accounting for CompactDropdown min/max constraints."""
        if hasattr(ctrl, '_calc_auto_width'):
            auto = ctrl._calc_auto_width()
            fixed = getattr(ctrl, '_fixed', None)
            mn = getattr(ctrl, '_min', 0) or 0
            mx = getattr(ctrl, '_max', 0) or 0
            if fixed is not None:
                return fixed
            if mn and auto < mn:
                return mn
            if mx and auto > mx:
                return mx
            return auto
        w = getattr(ctrl, "width", None) or 0
        if w:
            return w
        if hasattr(ctrl, '_fixed') and ctrl._fixed:
            return ctrl._fixed
        if hasattr(ctrl, '_label'):
            txt = ctrl._label.value or ""
            return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 34
        if hasattr(ctrl, '_value'):
            txt = str(ctrl._value or "")
            return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 34
        content = getattr(ctrl, "content", None)
        if content is not None:
            if hasattr(content, "value") and content.value:
                txt = content.value
            elif hasattr(content, "controls"):
                txt = "".join(c.value for c in content.controls if hasattr(c, "value") and c.value)
            else:
                txt = ""
            if txt:
                return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 40
        txt = getattr(ctrl, "label", None) or getattr(ctrl, "value", None) or ""
        return sum(13 if ord(ch) > 127 else 7 for ch in str(txt)) + 28

    def _fit_window_to_content(self, update=True, resize_window=False):
        main_width = self._estimate_main_panel_width()
        if getattr(self, "_main_panel", None) is not None:
            self._main_panel.width = main_width
            self._main_panel.expand = False
        log_panel = getattr(self, "log_panel", None)
        if log_panel is not None:
            if log_panel.visible:
                log_panel.width = None
                log_panel.expand = True
            else:
                log_panel.width = 0
                log_panel.expand = False
        width = self._estimate_window_width(main_width)
        self.page.window.min_width = main_width
        self.page.window.max_width = None
        if resize_window:
            self.page.window.width = width
        self.page.window.height = self._estimate_window_height()
        if resize_window:
            try:
                self.page.window.update()
            except RuntimeError:
                pass
            self.page.window.width = None
        if update:
            self.page.update()

    def _on_page_resize(self, e=None):
        pass

    def _spacing_width(self, controls, spacing):
        visible_count = sum(1 for ctrl in controls if self._control_has_effective_content(ctrl))
        return max(0, visible_count - 1) * (spacing or 0)

    def _estimate_control_width(self, ctrl):
        if ctrl is None or getattr(ctrl, "visible", True) is False:
            return 0
        if getattr(ctrl, "expand", None) and getattr(ctrl, "width", None) is None:
            content = getattr(ctrl, "content", None)
            if content is None:
                return 0
        if isinstance(ctrl, ft.Container):
            content = getattr(ctrl, "content", None)
            if content is not None:
                child_width = self._estimate_control_width(content)
                explicit = getattr(ctrl, "width", None) or 0
                padding = getattr(ctrl, "padding", None)
                pad_x = 0
                if padding is not None:
                    pad_x = (getattr(padding, "left", 0) or 0) + (getattr(padding, "right", 0) or 0)
                return max(explicit, child_width + pad_x)
        if isinstance(ctrl, ft.Row):
            controls = getattr(ctrl, "controls", []) or []
            return sum(self._estimate_control_width(c) for c in controls) + self._spacing_width(controls, getattr(ctrl, "spacing", 0))
        if isinstance(ctrl, ft.Column):
            controls = getattr(ctrl, "controls", []) or []
            return max((self._estimate_control_width(c) for c in controls), default=0)
        return self._ctrl_width(ctrl)

    def _estimate_group_width(self, container):
        if container is None or getattr(container, "visible", True) is False:
            return 0
        content = getattr(container, "content", None)
        content_width = self._estimate_control_width(content)
        padding = getattr(container, "padding", None)
        pad_x = 0
        if padding is not None:
            pad_x = (getattr(padding, "left", 0) or 0) + (getattr(padding, "right", 0) or 0)
        border_x = 2
        return content_width + pad_x + border_x

    def _estimate_main_panel_width(self):
        if not getattr(self, "depth_group", None):
            return S(696)
        sections = [self.lang_group, self.depth_group, self.device_group]
        widths = [self._estimate_group_width(section) for section in sections]
        if getattr(self, "stream_container", None) and self.stream_container.visible:
            widths.append(self._estimate_group_width(self.stream_container))
        return max(S(520), min(S(1040), max(widths + [0]) + S(12)))

    def _estimate_window_width(self, main_width=None):
        if not getattr(self, "depth_group", None):
            return S(696)
        content_width = self._estimate_main_panel_width() if main_width is None else main_width
        if getattr(self, "log_panel", None) and self.log_panel.visible:
            content_width += S(500)
        page_padding = (getattr(self.page, "padding", 0) or 0) * 2
        spacing = S(10) if getattr(self, "log_panel", None) and self.log_panel.visible else 0
        safety_margin = S(12)
        return content_width + page_padding + spacing + safety_margin

    def _control_has_effective_content(self, ctrl):
        if ctrl is None:
            return False
        if getattr(ctrl, "visible", True) is False:
            return False
        content = getattr(ctrl, "content", None)
        if content is not None:
            return self._control_has_effective_content(content)
        controls = getattr(ctrl, "controls", None)
        if controls is not None:
            return any(self._control_has_effective_content(child) for child in controls)
        return True

    def _estimate_group_height(self, container, include_margin=True):
        if container is None or getattr(container, "visible", True) is False:
            return 0
        content = getattr(container, "content", None)
        controls = getattr(content, "controls", None)
        if not controls:
            return 0
        visible_rows = sum(1 for ctrl in controls if self._control_has_effective_content(ctrl))
        if visible_rows <= 0:
            return 0
        row_height = S(34)
        row_spacing = getattr(content, "spacing", S(8)) or 0
        padding_v = S(24)
        border_v = 2
        margin_v = S(8) if include_margin else 0
        return padding_v + visible_rows * row_height + max(0, visible_rows - 1) * row_spacing + border_v + margin_v

    def _estimate_window_height(self):
        if not getattr(self, "depth_group", None):
            return S(768)
        scroll_spacing = getattr(getattr(self, "_scroll_area", None), "spacing", S(8)) or 0
        visible_sections = []
        for section in [self.lang_group, self.depth_group, self.device_group]:
            if section is not None and getattr(section, "visible", True):
                visible_sections.append(section)
        scroll_height = sum(self._estimate_group_height(section) for section in visible_sections)
        if getattr(self, "stream_container", None) and self.stream_container.visible:
            scroll_height += scroll_spacing
            scroll_height += self._estimate_group_height(self.stream_container, include_margin=False)
        page_padding = (getattr(self.page, "padding", 0) or 0) * 2
        footer_height = S(58)
        window_chrome = S(42)
        safety_margin = S(0)
        min_height = S(560)
        max_height = S(1040)
        estimated = scroll_height + footer_height + page_padding + window_chrome + safety_margin
        return max(min_height, min(max_height, estimated))

    # ── label alignment ──

    def _auto_align_labels(self, force=False):
        if self._labels_aligned and not force:
            return
        left_labels = [
            self.depth_model_label, self.depth_resolution_label, self.depth_quick_label,
            self.convergence_label, self.dynamic_convergence_label, self.depth_strength_label, self.depth_pop_label,
            self.foreground_pop_label, self.midground_pop_label, self.background_pop_label,
            self.antialiasing_label, self.stereo_preset_label, self.parallax_budget_label,
            self.scene_reset_label, self.edge_dilation_label, self.mask_feather_label, self.hole_fill_mode_label,
            self.depth_separation_label,
            self.acceleration_label, self.computing_device_label, self.capture_tool_label,
            self.target_fps_label, self.render_policy_label, self.render_fixed_label,
            self.render_min_dimension_label, self.run_mode_label,
            self.stereo_output_label, self.controller_label, self.lang_label,
            self.stream_url_label, self.stream_port_label,
            self.stream_proto_label, self.audio_label, self.crf_label,
        ]
        right_labels = [
            self.temporal_strength_label,
            self.edge_threshold_label, self.anaglyph_label,
            self.render_scale_label, self.render_max_pixels_label, self.render_align_label,
            self.display_mode_label, self.xr_headset_label, self.environment_label,
            self.theme_label, self.stream_quality_label, self.stream_key_label,
            self.audio_delay_label,
        ]

        def _est(t):
            return sum(S(12) if ord(c) > 127 else S(7) for c in t)

        all_labels = left_labels + right_labels
        max_w = max(_est(lbl.value) for lbl in all_labels)
        final_w = int(max_w * 1.15) + S(10)

        for lbl in all_labels:
            lbl.width = final_w

        self._label_max_width = final_w
        set_label_align_width(final_w)
        for inst in getattr(self, '_dropdowns', []):
            inst.reapply_width()
        if hasattr(self, '_row8_spacer'):
            capture_mode_w = self._ctrl_width(self.capture_mode_dd)
            self._row8_spacer.width = max(0, final_w - capture_mode_w - 1)
            self._safe_update(self._row8_spacer)
        if hasattr(self, '_accel_spacer'):
            self._accel_spacer.width = final_w
            self._safe_update(self._accel_spacer)
        self._labels_aligned = True

    # ── UI construction ──

    def build_ui(self):
        page = self.page
        page.controls.clear()
        self._dropdowns = []
        CompactDropdown._instances = self._dropdowns

        # Row 1: Depth model
        self.depth_model_label = ft.Text("Depth Model:", size=FONT_SIZE, width=S(130))
        default_family, default_size = parse_model_name(DEFAULT_MODEL_LIST[0]) if DEFAULT_MODEL_LIST else ("", "")
        self.depth_model_dd = CompactDropdown(
            options=[f for f in DEFAULT_FAMILIES],
            value=default_family,
            on_select=self.on_model_family_change,
            min_width=S(200), max_width=S(300))
        self.model_size_dd = CompactDropdown(
            options=FAMILY_TO_SIZES.get(default_family, []),
            value=default_size,
            on_select=self.on_model_size_change,
            width=S(110))
        self.fp16_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="FP16")
        row0 = ft.Row([
            self.depth_model_label, self.depth_model_dd,
            ft.Container(width=S(8)), self.model_size_dd,
        ], spacing=1)

        # Row 2: Depth resolution + Depth Quick
        self.depth_resolution_label = ft.Text("Depth Resolution:", size=FONT_SIZE, width=S(130))
        self.depth_res_dd = CompactDropdown(options=[], width=S(130))
        self.convergence_label = ft.Text("Convergence:", size=FONT_SIZE, width=S(130))
        conv_options = [f"{i / 100:.2f}" for i in range(-50, 101, 5)]
        self.convergence_dd = CompactDropdown(width=S(130),
            options=[v for v in conv_options], value="0.00",
            on_select=self.on_stereo_hot_param_change)
        self.dynamic_convergence_label = ft.Text("Dynamic Convergence:", size=FONT_SIZE, width=S(130))
        dynamic_conv_strength_options = [f"{i / 100:.2f}" for i in range(0, 101, 10)]
        self.dynamic_convergence_strength_dd = CompactDropdown(width=S(130),
            options=[v for v in dynamic_conv_strength_options], value="0.00",
            on_select=self.on_stereo_hot_param_change)
        self.depth_quick_label = ft.Text("Depth Quick:", size=FONT_SIZE, width=S(130))
        self.depth_quick_dd = CompactDropdown(
            options=["Soft", "Standard", "Enhanced"], value="Standard",
            on_select=self.on_depth_quick_change, width=S(130))
        row1 = ft.Row([
            self.depth_resolution_label, self.depth_res_dd,
            ft.Container(width=S(40)), self.depth_quick_label, self.depth_quick_dd,
        ], spacing=1)

        # Row 3: Convergence + Depth Strength
        self.depth_strength_label = ft.Text("Depth Strength:", size=FONT_SIZE, width=S(130))
        ds_options = [f"{i / 100:.2f}" for i in range(0, 51, 5)]
        self.depth_strength_dd = CompactDropdown(width=S(130),
            options=[v for v in ds_options], value="0.25",
            on_select=self.on_stereo_hot_param_change)
        convergence_depth_row = ft.Row([
            self.convergence_label, self.convergence_dd,
            ft.Container(width=S(40)), self.dynamic_convergence_label, self.dynamic_convergence_strength_dd,
        ], spacing=1)
        depth_strength_row = ft.Row([self.depth_strength_label, self.depth_strength_dd], spacing=1)

        # Row 3b: Depth Pop + anti-aliasing
        self.depth_pop_label = ft.Text("Depth Pop:", size=FONT_SIZE, width=S(130))
        depth_pop_options = [f"{i / 10:.1f}" for i in range(-9, 0)] + [f"{i / 2:.1f}" for i in range(0, 11)]
        self.depth_pop_dd = CompactDropdown(width=S(130),
            options=[v for v in depth_pop_options], value="0.0",
            on_select=self.on_stereo_hot_param_change)
        self.antialiasing_label = ft.Text("Anti-aliasing:", size=FONT_SIZE, width=S(130))
        aa_options = [str(i) for i in range(11)]
        self.antialiasing_dd = CompactDropdown(width=S(130),
            options=[v for v in aa_options], value="2",
            on_select=self.on_stereo_hot_param_change)
        row2b = ft.Row([
            self.depth_pop_label, self.depth_pop_dd,
            ft.Container(width=S(40)), self.antialiasing_label, self.antialiasing_dd,
        ], spacing=1)

        # Row 4: Stereo runtime mode + parallax budget.
        self.stereo_preset_label = ft.Text("Stereo Mode:", size=FONT_SIZE, width=S(130))
        self.stereo_preset_dd = CompactDropdown(
            options=["Traditional / Fastest", "Cinema", "Game / Low Latency", "Image  / High Quality"],
            value="Cinema", width=S(130), on_select=self.on_stereo_preset_change)
        self.stereo_quality_label = ft.Text("Synthetic View:", size=FONT_SIZE, width=0, visible=False)
        self.stereo_quality_dd = CompactDropdown(options=self._stereo_quality_options(),
            value=self._stereo_quality_to_display("quality_4k"), width=S(1))
        self.stereo_quality_dd.visible = False
        self.parallax_budget_label = ft.Text("Parallax Budget:", size=FONT_SIZE, width=S(130))
        self.parallax_budget_dd = CompactDropdown(options=self._parallax_budget_options(),
            value=self._parallax_budget_to_display("standard"), width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row0 = ft.Row([self.stereo_preset_label, self.stereo_preset_dd,
            ft.Container(width=S(40)), self.parallax_budget_label, self.parallax_budget_dd], spacing=1)

        self.hole_fill_mode_label = ft.Text("Hole Fill Mode:", size=FONT_SIZE, width=S(130))
        self.hole_fill_mode_dd = CompactDropdown(
            options=self._hole_fill_mode_options(),
            value=self._hole_fill_mode_to_display("balanced"), width=S(130), on_select=self.on_stereo_hot_param_change)
        self.depth_separation_label = ft.Text("Depth Separation:", size=FONT_SIZE, width=S(130))
        self.depth_separation_dd = CompactDropdown(
            options=self._depth_separation_options(),
            value=self._depth_separation_to_display("standard"), width=S(130), on_select=self.on_depth_separation_change)
        self.advanced_stereo_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="Advanced Stereo", value=False, on_change=self.on_advanced_stereo_change)
        hole_fill_row = ft.Row([self.hole_fill_mode_label, self.hole_fill_mode_dd,
            ft.Container(width=S(40)), self.depth_separation_label, self.depth_separation_dd], spacing=1)
        advanced_stereo_row = ft.Row([ft.Container(width=S(130)), self.advanced_stereo_cb], spacing=1)

        self.temporal_strength_label = ft.Text("Temporal Strength:", size=FONT_SIZE, width=S(130))
        self.temporal_strength_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(0, 11)],
            value="0.7", width=S(130), on_select=self.on_stereo_hot_param_change)

        self.scene_reset_label = ft.Text("Scene Threshold:", size=FONT_SIZE, width=S(130))
        self.scene_reset_dd = CompactDropdown(options=["0.00", "0.12", "0.18", "0.22", "0.28", "0.35"],
            value="0.22", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row1 = ft.Row([self.temporal_strength_label, self.temporal_strength_dd,
            ft.Container(width=S(40)), self.scene_reset_label, self.scene_reset_dd], spacing=1)

        self.edge_dilation_label = ft.Text("Edge Dilation:", size=FONT_SIZE, width=S(130))
        self.edge_dilation_dd = CompactDropdown(options=["0", "1", "2", "3", "4"],
            value="2", width=S(130), on_select=self.on_stereo_hot_param_change)
        self.mask_feather_label = ft.Text("Mask Feather:", size=FONT_SIZE, width=S(130))
        self.mask_feather_dd = CompactDropdown(options=["0", "1", "2", "3", "4", "5"],
            value="3", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row3 = ft.Row([self.edge_dilation_label, self.edge_dilation_dd,
            ft.Container(width=S(40)), self.mask_feather_label, self.mask_feather_dd], spacing=1)
        self.edge_threshold_label = ft.Text("Edge Threshold:", size=FONT_SIZE, width=S(130))
        self.edge_threshold_dd = CompactDropdown(options=[f"{i / 100:.2f}" for i in range(0, 11)],
            value="0.04", width=S(130), on_select=self.on_stereo_hot_param_change)

        pop_options = [f"{i / 100:.2f}" for i in range(50, 161, 5)]
        self.foreground_pop_label = ft.Text("Foreground Pop:", size=FONT_SIZE, width=S(130))
        self.foreground_pop_dd = CompactDropdown(options=pop_options, value="1.00", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row3b = ft.Row([self.edge_threshold_label, self.edge_threshold_dd,
            ft.Container(width=S(40)), self.foreground_pop_label, self.foreground_pop_dd], spacing=1)
        self.midground_pop_label = ft.Text("Midground Pop:", size=FONT_SIZE, width=S(130))
        self.midground_pop_dd = CompactDropdown(options=pop_options, value="1.00", width=S(130), on_select=self.on_stereo_hot_param_change)
        self.background_pop_label = ft.Text("Background Pop:", size=FONT_SIZE, width=S(130))
        self.background_pop_dd = CompactDropdown(options=pop_options, value="1.00", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row3c = ft.Row([
            self.midground_pop_label, self.midground_pop_dd,
            ft.Container(width=S(40)), self.background_pop_label, self.background_pop_dd,
        ], spacing=1)

        self.cross_eyed_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="Cross Eyed", value=False, on_change=self.on_stereo_hot_param_change)
        self.anaglyph_label = ft.Text("Anaglyph:", size=FONT_SIZE, width=S(130))
        self.anaglyph_dd = CompactDropdown(options=["red_cyan", "green_magenta", "amber_blue"],
            value="red_cyan", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row4 = ft.Row([self.anaglyph_label, self.anaglyph_dd,
            ft.Container(width=S(40)), self.cross_eyed_cb, ft.Container(width=S(20)), self.fp16_cb], spacing=1)

        self._advanced_stereo_rows = [convergence_depth_row, depth_strength_row, row2b, stereo_row1, stereo_row3, stereo_row3b, stereo_row3c, stereo_row4]

        # Acceleration group
        self.acceleration_label = ft.Text("Acceleration:", size=FONT_SIZE, width=S(130))
        self.torch_compile_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="torch.compile")
        self.tensorrt_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="TensorRT", on_change=self._on_trt_toggle)
        self.coreml_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="CoreML", on_change=self._on_coreml_toggle)
        self.openvino_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="OpenVINO", on_change=self._on_openvino_toggle)
        self.migraphx_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="MIGraphX", on_change=self._on_migraphx_toggle)
        self.recompile_trt_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile TensorRT")
        self.recompile_coreml_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile CoreML")
        self.recompile_openvino_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile OpenVINO")
        self.recompile_migraphx_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile MIGraphX")
        accel_row1 = ft.Row([self.torch_compile_cb, self.tensorrt_cb, self.recompile_trt_cb], spacing=S(20))
        accel_row2 = ft.Row([self.coreml_cb, self.recompile_coreml_cb, self.openvino_cb, self.recompile_openvino_cb], spacing=S(20))
        accel_row3 = ft.Row([self.migraphx_cb, self.recompile_migraphx_cb], spacing=S(20))
        self._accel_spacer = ft.Container(width=0)
        self.row4a = ft.Row([self.acceleration_label, accel_row1], spacing=1)
        self.row4b = ft.Row([self._accel_spacer, accel_row2], spacing=1)
        self.row4c = ft.Row([self._accel_spacer, accel_row3], spacing=1)
        self._advanced_stereo_rows.extend([self.row4a, self.row4b, self.row4c])
        for row in self._advanced_stereo_rows:
            row.visible = self.advanced_stereo_cb.value

        # Row 6: Computing device
        self.computing_device_label = ft.Text("Computing Device:", size=FONT_SIZE, width=S(130))
        device_names = [v["name"] for v in DEVICES.values()]
        self.device_dd = CompactDropdown(options=[n for n in device_names],
            on_select=self.on_device_change, min_width=S(180))
        self.showfps_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Show FPS")
        self.local_vsync_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="VSync", value=DEFAULTS.get("VSync", False))
        self.target_fps_label = ft.Text("Capture FPS:", size=FONT_SIZE, width=S(130))
        self.target_fps_dd = CompactDropdown(options=["Auto", "60", "72", "80", "90", "120"],
            value="Auto", width=S(74))
        self.xr_preview_cb = ft.Checkbox(label="XR Preview Window",
            value=DEFAULTS.get("XR Preview Window", True))
        self.advanced_device_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT,
            label="Advanced Options", value=False, on_change=self.on_advanced_device_change)
        row5 = ft.Row([self.computing_device_label, self.device_dd,
            ft.Container(width=S(15)), self.advanced_device_cb], spacing=1)

        # Row 7: Capture tool
        self.capture_tool_label = ft.Text("Capture Tool:", size=FONT_SIZE, width=S(130))
        ct_options = get_capture_tool_options(DEVICES.get(0, {}).get("name", ""))
        self.capture_tool_dd = CompactDropdown(options=[o for o in ct_options],
            on_select=self.on_capture_tool_change, min_width=S(160))
        row6 = ft.Row([self.capture_tool_label, self.capture_tool_dd,
            ft.Container(width=S(15)), self.showfps_cb], spacing=1)
        self.row6b = ft.Row([self.target_fps_label, self.target_fps_dd,
            ft.Container(width=S(20)), self.xr_preview_cb, ft.Container(width=S(20)), self.local_vsync_cb], spacing=1)
        self.render_policy_label = ft.Text("Render Policy:", size=FONT_SIZE, width=S(130), visible=False)
        self.render_policy_dd = CompactDropdown(
            options=["Scaled"], value="Scaled", width=S(130),
            on_select=self.on_render_policy_change)
        self.render_policy_dd.visible = False
        self.render_scale_label = ft.Text("Render Scale:", size=FONT_SIZE, width=S(130))
        self.render_scale_dd = CompactDropdown(options=self._render_scale_options(),
            value=self._render_scale_to_display(DEFAULTS["Render Scale"]), width=S(130))
        self.render_align_label = ft.Text("Render Align:", size=FONT_SIZE, width=S(130))
        self.render_align_dd = CompactDropdown(options=["1", "8", "16", "32"],
            value=str(DEFAULTS["Render Align"]), width=S(130))
        self.row6d = ft.Row([self.render_scale_label, self.render_scale_dd,
            ft.Container(width=S(40)), self.render_align_label, self.render_align_dd], spacing=1)
        self.render_fixed_label = ft.Text("Render Fixed Size:", size=FONT_SIZE, width=S(130), visible=False)
        self.render_fixed_dd = CompactDropdown(
            options=["1280x720", "1600x900", "1920x1080", "2560x1440", "3840x2160"],
            value="1920x1080", width=S(130))
        self.render_fixed_dd.visible = False
        self.row6e = ft.Row([self.render_policy_label, self.render_policy_dd,
            self.render_fixed_label, self.render_fixed_dd], spacing=1, visible=False)
        self.render_max_pixels_label = ft.Text("Render Pixel Cap:", size=FONT_SIZE, width=S(130))
        self.render_max_pixels_dd = CompactDropdown(
            options=["921600", "2073600", "3686400", "8294400"], value="8294400", width=S(130))
        self.render_min_dimension_label = ft.Text("Render Min Side:", size=FONT_SIZE, width=S(130))
        self.render_min_dimension_dd = CompactDropdown(options=["360", "480", "540", "720"],
            value="480", width=S(130))
        self.row6f = ft.Row([self.render_min_dimension_label, self.render_min_dimension_dd,
            ft.Container(width=S(40)), self.render_max_pixels_label, self.render_max_pixels_dd], spacing=1)
        self.upscaler_label = ft.Text("", size=FONT_SIZE, width=0, visible=False)
        self.upscaler_dd = CompactDropdown(options=["Off"], value="Off", width=S(1))
        self.upscaler_dd.visible = False
        self.upscaler_sharpness_label = ft.Text("", size=FONT_SIZE, width=0, visible=False)
        self.upscaler_sharpness_dd = CompactDropdown(options=["0.00"], value="0.00", width=S(1))
        self.upscaler_sharpness_dd.visible = False
        self.row6c = ft.Row([], spacing=1, visible=False)
        if OS_NAME == "Linux":
            self.capture_tool_label.visible = False
            self.capture_tool_dd.visible = False

        # Row 8: Run mode + Display mode / Controller
        self.run_mode_label = ft.Text("Run Mode:", size=FONT_SIZE, width=S(130))
        self.run_mode_dd = CompactDropdown(on_select=self.on_run_mode_change, width=S(130))
        self.xr_headset_label = ft.Text("Headset Model:", size=FONT_SIZE, width=S(130))
        self.xr_headset_dd = CompactDropdown(
            options=xr_headset_options(self.locale),
            value=xr_headset_to_display(DEFAULTS.get("XR Headset Model"), self.locale),
            on_select=self.on_xr_headset_change,
            width=S(130))
        self.display_mode_label = ft.Text("Display Mode:", size=FONT_SIZE, width=S(130))
        self.display_mode_dd = CompactDropdown(
            options=["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB", "Depth Map", "Anaglyph", "Interleaved", "Mono", "Leia"],
            value="Half-SBS", width=S(130))
        self.controller_label = ft.Text("Controller:", size=FONT_SIZE, width=S(130))
        try:
            ctrl_base = os.path.join(BASE_DIR, "xr_viewer", "controllers")
            ctrl_dirs = sorted(d for d in os.listdir(ctrl_base) if os.path.isdir(os.path.join(ctrl_base, d)))
        except (FileNotFoundError, OSError):
            ctrl_dirs = []
        if not ctrl_dirs:
            ctrl_dirs = ["PICO"]
        self.ctrl_model_dd = CompactDropdown(options=[c for c in ctrl_dirs], value="PICO", width=S(130))
        self.environment_label = ft.Text("Environment:", size=FONT_SIZE, width=S(130))
        self.env_model_keys = get_environment_model_options(return_keys=True)
        self.env_model_display_names = load_environment_display_names(self.env_model_keys)
        env_options = get_environment_model_options(self.locale)
        self.env_key = DEFAULTS.get("Environment Model", "Default")
        if str(self.env_key).strip().lower() == "none":
            self.env_key = "Default"
        if self.env_key not in self.env_model_keys:
            self.env_key = self.env_model_keys[0] if self.env_model_keys else "Default"
        self.env_model_dd = CompactDropdown(
            options=[e for e in env_options],
            value=environment_display_label(self.env_key, self.locale, self.env_model_display_names),
            on_select=self.on_env_change,
            width=S(130))
        self.row7a = ft.Row([self.run_mode_label, self.run_mode_dd, ft.Container(width=S(40)),
            self.xr_headset_label, self.xr_headset_dd,
            self.display_mode_label, self.display_mode_dd], spacing=1)
        self.row7b = ft.Row([self.controller_label, self.ctrl_model_dd, ft.Container(width=S(40)),
            self.environment_label, self.env_model_dd], spacing=1)

        # Row 9: Input monitor/window + Refresh
        self.capture_mode_dd = CompactDropdown(options=["Monitor", "Window"],
            value="Monitor", on_select=self.on_capture_mode_change, width=S(100))
        self.monitor_dd = CompactDropdown(on_select=self._on_monitor_change, max_width=S(300))
        self.window_dd = CompactDropdown(on_select=self.on_window_selected, max_width=S(300))
        self.refresh_btn = ft.Button(content=ft.Text("Refresh", size=FONT_SIZE),
            width=S(130), on_click=self.refresh_monitor_and_window)
        self._row8_spacer = ft.Container(width=S(60))
        row8 = ft.Row([self.capture_mode_dd, self._row8_spacer, self.monitor_dd, self.window_dd,
            ft.Container(width=S(8)), ft.Container(expand=True), self.refresh_btn], spacing=1)

        # Row 10: Stereo output + checkboxes
        self.stereo_output_label = ft.Text("Stereo Output:", size=FONT_SIZE, width=S(130))
        self.stereo_monitor_dd = CompactDropdown(options=[],
            on_select=lambda e: self._fit_window_to_content())
        self.fill_16_9_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Fill 16:9")
        self.fix_aspect_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Fix Viewer Aspect")
        self.lossless_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="LSFG")
        self._stereo_spacer = ft.Container(width=S(10))
        self.row9 = ft.Row([self.stereo_output_label, self.stereo_monitor_dd, self._stereo_spacer,
            ft.Row([self.fill_16_9_cb, self.fix_aspect_cb, self.lossless_cb], spacing=S(20))], spacing=1)

        # Bottom: Language + Theme + Buttons
        self.lang_label = ft.Text("Set Language:", size=FONT_SIZE, width=S(130))
        self.lang_dd = CompactDropdown(options=["English", "简体中文"],
            value="English", on_select=self.on_language_change, width=S(130))
        self.theme_label = ft.Text("Theme:", size=FONT_SIZE, width=S(130))
        self.theme_dd = CompactDropdown(
            options=["system", "blue", "green", "red", "purple", "orange", "teal", "pink", "grey"],
            value="system", on_select=self.on_theme_change, width=S(130))
        self.reset_btn = ft.Button(content=ft.Text("Reset", size=FONT_SIZE),
            width=S(130), on_click=self.reset_defaults)
        self.stop_btn = ft.Button(content=ft.Text("Stop", size=FONT_SIZE),
            width=S(130), on_click=self.stop_process)
        self.run_btn = ft.Button(content=ft.Text("Run", size=FONT_SIZE),
            width=S(150), on_click=self.save_and_run)
        lang_row = ft.Row([self.lang_label, self.lang_dd, ft.Container(width=S(40)),
            self.theme_label, self.theme_dd], spacing=1)

        self.status_text = ft.Text("", italic=True, size=FONT_SIZE)
        self.log_visibility_link = ft.Text(
            UI_MESSAGES[self.locale].get("Hide log panel link", "Hide log window ->"),
            size=FONT_SIZE,
            color=ft.Colors.BLUE,
        )
        self.log_visibility_link_box = ft.Container(
            content=self.log_visibility_link,
            padding=ft.Padding(S(8), S(4), S(8), S(4)),
            on_click=self.on_log_visibility_link,
        )

        # Assembly
        depth_group = ft.Container(
            ft.Column([row0, row1, stereo_row0, hole_fill_row, advanced_stereo_row,
                       convergence_depth_row, depth_strength_row, row2b, stereo_row1, stereo_row3, stereo_row3b,
                       stereo_row3c, stereo_row4, self.row4a, self.row4b, self.row4c], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE),
                             ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)))
        device_group = ft.Container(
            ft.Column([row5, row6, self.row6b, self.row6d, self.row6e, self.row6f,
                       self.row7a, self.row7b, row8, self.row6c, self.row9], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE),
                             ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)))
        lang_group = ft.Container(
            ft.Column([lang_row], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE),
                             ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)))
        self.lang_group = lang_group
        self.depth_group = depth_group
        self.device_group = device_group
        self._build_streamer_rows()

        scroll_area = ft.Column([
            self.lang_group, self.depth_group, self.device_group, self.stream_container,
        ], scroll=ft.ScrollMode.AUTO, expand=False, tight=True, spacing=S(8))
        self.log_level_dd = CompactDropdown(
            options=["ALL", "STATUS", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            value="ALL",
            width=S(110),
            on_select=self.on_log_level_filter,
        )
        self.report_issue_btn = ft.Button(
            content=ft.Text(UI_MESSAGES[self.locale].get("Report issue", "Report bug"), size=FONT_SIZE),
            width=S(150),
            on_click=self.on_report_issue,
        )
        self.open_log_file_btn = ft.Button(
            content=ft.Text(UI_MESSAGES[self.locale].get("Open log file", "Open log"), size=FONT_SIZE),
            width=S(150),
            on_click=self.on_open_log_file,
        )
        self.download_progress_title = ft.Text("Download", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE)
        self.download_progress_percent = ft.Text("0%", size=12, color=ft.Colors.BLUE)
        self.download_progress_bar = ft.ProgressBar(value=0, height=S(8), color=ft.Colors.BLUE, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST)
        self.download_progress_detail = ft.Text("", size=11, color=ft.Colors.GREY, no_wrap=True, overflow=ft.TextOverflow.VISIBLE)
        self.download_progress_panel = ft.Container(
            content=ft.Column([
                ft.Row([self.download_progress_title, ft.Container(expand=True), self.download_progress_percent]),
                self.download_progress_bar,
                self.download_progress_detail,
            ], spacing=S(4)),
            padding=ft.Padding(S(8), S(6), S(8), S(6)),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=6,
            visible=False,
        )
        self.log_text = ft.Text(
            spans=[],
            selectable=True,
            no_wrap=True,
            overflow=ft.TextOverflow.VISIBLE,
            size=12,
        )
        self.log_scroll_row = ft.Row(
            [self.log_text],
            scroll=ft.Scrollbar(orientation=ft.ScrollbarOrientation.BOTTOM),
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        self.log_viewport = ft.Column(
            [self.log_scroll_row],
            scroll=ft.Scrollbar(orientation=ft.ScrollbarOrientation.RIGHT),
            auto_scroll=True,
            expand=True,
            tight=True,
            spacing=0,
        )
        self.log_body = ft.Container(
            content=self.log_viewport,
            expand=True,
            visible=True,
        )
        self.log_panel = ft.Container(
            content=ft.Column([
                ft.Row([
                    self.report_issue_btn, self.open_log_file_btn, ft.Container(expand=True),
                    self.log_level_dd,
                ], spacing=S(6), vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.download_progress_panel,
                self.log_body,
            ], spacing=S(6), expand=True),
            visible=False,
            expand=True,
            padding=ft.Padding(S(10), S(10), S(10), S(10)),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE),
                             ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6,
        )


        btn_row = ft.Row([self.reset_btn, ft.Container(expand=True),
            ft.Container(content=ft.Row([self.stop_btn, self.run_btn], spacing=S(20)),
                         padding=ft.Padding(0, 0, S(10), 0))])
        self._btn_bar = ft.Container(content=btn_row)
        self._status_bar = ft.Row([
            ft.Container(content=self.status_text, bgcolor=ft.Colors.SURFACE_CONTAINER,
                         border_radius=0, padding=ft.Padding(S(8), S(4), S(8), S(4)), expand=True),
            self.log_visibility_link_box])
        footer = ft.Container(
            ft.Column([self._btn_bar, self._status_bar], spacing=S(6)),
            padding=ft.Padding(0, S(6), 0, 0))
        self._scroll_area = scroll_area
        self._footer = footer
        self._main_panel = ft.Container(
            content=ft.Column([scroll_area, footer], expand=True, tight=True, spacing=0),
            expand=False,
        )
        self._root_row = ft.Row([self._main_panel, self.log_panel], expand=True, tight=True, spacing=S(10))
        page.add(self._root_row)

    # ── streamer rows ──

    def _build_streamer_rows(self):
        self.stream_url_label = ft.Text("Stream URL:", size=FONT_SIZE, width=S(150))
        self.stream_url_tf = ft.Container(
            content=ft.Row([ft.Text("", size=FONT_SIZE)], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            height=S(32), padding=ft.Padding(S(8), 0, S(8), 0), expand=True,
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE),
                             ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=4, on_click=self.copy_url_to_clipboard)
        self.preview_btn = ft.Button(content=ft.Text("Preview", size=FONT_SIZE),
            width=S(130), on_click=self.preview_in_browser)
        self.stream_url_row = ft.Row(
            [self.stream_url_label, self.stream_url_tf, ft.Container(width=S(10)), self.preview_btn], spacing=2)
        self.stream_port_label = ft.Text("Streamer Port:", size=FONT_SIZE, width=S(150))
        self.stream_port_tf = CompactTextField(value=str(DEFAULT_PORT), width=S(130),
            on_change=self.update_stream_url, filter=r"[0-9]", max_length=5)
        self.stream_quality_label = ft.Text("Stream Quality:", size=FONT_SIZE)
        qual_vals = [str(i) for i in range(100, 49, -5)]
        self.stream_quality_dd = CompactDropdown(width=S(130), options=[q for q in qual_vals], value="100")
        self.stream_port_quality_row = ft.Row(
            [self.stream_port_label, self.stream_port_tf, ft.Container(width=S(40)),
             self.stream_quality_label, self.stream_quality_dd], spacing=1)
        self.stream_proto_label = ft.Text("Stream Protocol:", size=FONT_SIZE, width=S(150))
        self.stream_proto_dd = CompactDropdown(width=S(130),
            options=["RTMP", "RTSP", "HLS", "HLS M3U8", "WebRTC"],
            value="HLS", on_select=self._on_stream_protocol_change)
        self.stream_key_label = ft.Text("Stream Key:", size=FONT_SIZE, width=S(130))
        self.stream_key_tf = CompactTextField(value="live", width=S(130),
            on_change=self._on_stream_key_change)
        self.stream_proto_row = ft.Row([self.stream_proto_label, self.stream_proto_dd,
            ft.Container(width=S(40)), self.stream_key_label, self.stream_key_tf], spacing=1)
        self.audio_label = ft.Text("Stereo Mix:", size=FONT_SIZE, width=S(150))
        self.audio_dd = CompactDropdown(options=[], min_width=S(130))
        self.audio_row = ft.Row([self.audio_label, self.audio_dd], spacing=1)
        self.crf_label = ft.Text("CRF:", size=FONT_SIZE, width=S(150))
        self.crf_tf = CompactTextField(value="20", width=S(130), filter=r"[0-9]", max_length=2)
        self.audio_delay_label = ft.Text("Audio Delay (s):", size=FONT_SIZE, width=S(130))
        self.audio_delay_tf = CompactTextField(value="-0.15", width=S(130), filter=r"[0-9\-\.]", max_length=6)
        self.crf_row = ft.Row([self.crf_label, self.crf_tf, ft.Container(width=S(40)),
            self.audio_delay_label, self.audio_delay_tf], spacing=1)
        self._streamer_rows = [
            self.stream_url_row, self.stream_port_quality_row, self.stream_proto_row,
            self.crf_row, self.audio_row,
        ]
        self.stream_container = ft.Container(
            ft.Column([], spacing=S(8)), visible=False,
            padding=ft.Padding(S(16), S(10), S(16), S(10)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE),
                             ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6)

    def _show_streamer_rows(self, *row_indices):
        col = self.stream_container.content.controls
        col.clear()
        for i in row_indices:
            if 0 <= i < len(self._streamer_rows):
                col.append(self._streamer_rows[i])
        self.stream_container.visible = bool(row_indices)
        self.stream_container.update()
        self._fit_window_to_content()

    @staticmethod
    def _get_streamer_row_map():
        return {
            "Local Viewer": [], "3D Monitor": [], "OpenXR Link": [],
            "MJPEG Streamer": [0, 1], "Legacy Streamer": [0, 1],
            "RTMP Streamer": [0, 1, 2, 3, 4],
        }

    # ── data population ──

    def populate_monitors(self):
        self.monitor_label_to_index = {}
        monitors = list_monitors()
        if not monitors:
            self.monitor_dd.options = []
            self.monitor_dd.update()
            return {}
        primary_index = get_primary_monitor_index()
        current_val = self.monitor_dd.value if hasattr(self, 'monitor_dd') else ""
        found = False
        opts = []
        for mon in monitors:
            capture_index = mon["capture_index"]
            display_number = mon["display_number"]
            is_primary = capture_index == primary_index
            suffix = PRIMARY_MONITOR_SUFFIX if is_primary else ""
            label = f"{display_number}: {mon['width']}x{mon['height']} @ ({mon['left']},{mon['top']}){suffix}"
            self.monitor_label_to_index[label] = capture_index
            opts.append(label)
            if label == current_val:
                found = True
        self.monitor_dd.options = opts
        if found:
            self.monitor_dd.value = current_val
        else:
            primary_label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == primary_index), None)
            self.monitor_dd.value = primary_label or (list(self.monitor_label_to_index.keys())[0] if self.monitor_label_to_index else "")
        self.monitor_dd.update()
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        return self.monitor_label_to_index

    def populate_devices(self):
        self.device_label_to_index = {}
        device_dict = DEVICES
        opts = []
        for idx, dev_info in device_dict.items():
            label = dev_info["name"]
            self.device_label_to_index[label] = idx
            opts.append(label)
        self.device_dd.options = opts
        default_idx = DEFAULTS.get("Computing Device", 0)
        default_label = next((lbl for lbl, i in self.device_label_to_index.items() if i == default_idx), None)
        self.device_dd.value = default_label or (opts[0] if opts else "")
        self.device_dd.update()
        return self.device_label_to_index

    def _apply_stereo_output(self, cfg):
        mon_count = self._get_monitor_count()
        if mon_count <= 1:
            self.stereo_monitor_dd.value = "Viewer Window"
            return
        saved = cfg.get("Stereo Output")
        input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
        if saved is not None:
            label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == saved), None)
            if label and label != input_label:
                self.stereo_monitor_dd.value = label
                return
        fallback = None
        for lbl in self.monitor_label_to_index:
            if lbl != input_label:
                fallback = lbl
                break
        self.stereo_monitor_dd.value = fallback if fallback else "Viewer Window"

    @staticmethod
    def _get_monitor_count():
        try:
            import mss
            with mss.mss() as sct:
                return len(sct.monitors) - 1
        except Exception:
            return 0

    def update_stereo_monitor_menu(self):
        if not hasattr(self, 'stereo_monitor_dd'):
            return
        input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
        opts = ["Viewer Window"]
        for label in self.monitor_label_to_index:
            if label != input_label:
                opts.append(label)
        current = self.stereo_monitor_dd.value
        valid = current in opts
        self.stereo_monitor_dd.options = opts
        if not valid:
            self.stereo_monitor_dd.value = opts[0] if opts else "Viewer Window"
        self.stereo_monitor_dd.update()

    def update_depth_resolution_options(self, model_name):
        resolutions = ALL_MODELS.get(model_name, {}).get("resolutions", [DEFAULTS["Depth Resolution"]])
        self.depth_res_dd.options = [str(r) for r in resolutions]
        cur = self.depth_res_dd.value
        if cur and cur in [str(r) for r in resolutions]:
            return
        preferred = 512 if "infinidepth" in str(model_name or "").lower() else DEFAULTS["Depth Resolution"]
        try:
            cur_num = int(cur) if cur else preferred
        except (ValueError, TypeError):
            cur_num = preferred
        closest = min(resolutions, key=lambda x: abs(x - cur_num))
        self.depth_res_dd.value = str(closest)
        self.depth_res_dd.update()
