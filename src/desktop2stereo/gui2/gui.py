"""Isolated GUI2 shell around the existing Desktop2Stereo runtime behavior.

The legacy GUI implementation is not edited. GUI2 reuses its mature config,
device, logging, and process lifecycle through a thin subclass while owning a
new menu/navigation shell and all new presentation code.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import flet as ft

from gui.gui import Desktop2StereoGUI
from gui import devices as devices_module
from gui.config import DEFAULTS
from gui.paths import BASE_DIR
from utils import VERSION

from .community import QQ_GROUP_NUMBER, QQ_INVITE_URL, WEBSITE_URL, qr_asset_path
from .localization import gui2_text
from .menu_registry import MenuItemSpec, build_menu_specs


PAGE_KEYS = (
    "home", "stereo", "quality", "performance", "advanced", "logs", "help",
)
GUI2_NAV_COLLAPSED_WIDTH = 72
GUI2_NAV_MIN_EXPANDED_WIDTH = 176
GUI2_NAV_MAX_EXPANDED_WIDTH = 320
GUI2_NAV_EXPAND_DELAY_SECONDS = 3.0
GUI2_NAV_COLLAPSE_DELAY_SECONDS = 1.0
PAGE_DESCRIPTION_KEYS = {
    "home": "home_description",
    "stereo": "stereo_title",
    "quality": "quality_title",
    "performance": "performance_title",
    "streaming": "streaming_title",
    "advanced": "advanced_title",
    "logs": "logs_title",
    "help": "help_title",
}


class Desktop2StereoGUI2(Desktop2StereoGUI):
    """GUI2 presentation shell with the legacy runtime contract intact."""

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self._gui2_page_index = 0
        self._gui2_pages: dict[str, ft.Control] = {}
        self._gui2_nav: ft.NavigationRail | None = None
        self._gui2_page_title: ft.Text | None = None
        self._gui2_page_description: ft.Text | None = None
        self._gui2_status: ft.Text | None = None
        self._gui2_language_label: ft.Text | None = None
        self._gui2_theme_label: ft.Text | None = None
        self._gui2_menu: ft.MenuBar | None = None
        self._gui2_nav_host: ft.Container | None = None
        self._gui2_nav_stack: ft.Stack | None = None
        self._gui2_compact = False
        self._gui2_nav_hovered = False
        self._gui2_nav_pointer_inside = False
        self._gui2_nav_hover_generation = 0
        self._gui2_nav_hover_task = None
        self._gui2_nav_expanded_width = GUI2_NAV_MIN_EXPANDED_WIDTH
        self._gui2_log_requested = False
        self._gui2_ready = False

    async def setup(self):
        await super().setup()
        self._gui2_ready = True

    def build_ui(self):
        # Build every legacy control first. The legacy root is removed before
        # the same control rows are mounted in the GUI2 page containers.
        super().build_ui()
        legacy_root = self._root_row
        try:
            self.page.remove(legacy_root)
        except (ValueError, RuntimeError):
            if legacy_root in self.page.controls:
                self.page.controls.remove(legacy_root)

        for control in (self.lang_group, self._footer):
            if control in self._scroll_area.controls:
                self._scroll_area.controls.remove(control)
        self.log_panel.visible = True

        depth_rows = list(self.depth_group.content.controls)
        device_rows = list(self.device_group.content.controls)
        self.depth_group.content.controls.clear()
        self.device_group.content.controls.clear()

        self._gui2_acceleration_rows = depth_rows[-3:]
        self._gui2_advanced_rows = depth_rows[5:-3]
        self._gui2_performance_rows = [
            device_rows[index] for index in (0, 1, 7, 8, 15)
            if index < len(device_rows)
        ] + self._gui2_acceleration_rows
        self._gui2_quality_rows = [
            device_rows[index] for index in (2, 3, 4, 5, 6)
            if index < len(device_rows)
        ]
        self._sync_advanced_stereo_visibility()
        self._apply_gui2_acceleration_policy()
        self._apply_gui2_performance_visibility()
        # GUI2 gives colour and LOD controls their own always-visible page.
        # Keep them independent from the legacy "advanced device" checkbox.
        for row in self._gui2_quality_rows:
            row.visible = True

        self._gui2_pages = {
            "home": self._build_home_page(device_rows),
            "stereo": self._section_page(
                "stereo_title",
                depth_rows[:5] + [self._build_advanced_stereo_group()],
                "Stereo",
            ),
            "quality": self._build_quality_page(),
            "performance": self._section_page(
                "performance_title", self._gui2_performance_rows, "Performance",
            ),
            "advanced": self._build_advanced_shortcut_page(),
            "logs": self._section_page("logs_title", [self.log_panel], "Logs"),
            "help": self._build_help_page(),
        }

        self._gui2_page_title = ft.Text(size=20, weight=ft.FontWeight.BOLD)
        self._gui2_page_description = ft.Text(size=12, color=ft.Colors.GREY)
        self._gui2_status = ft.Text(size=12, color=ft.Colors.GREY)
        self._gui2_language_label = ft.Text(size=12, data="footer_language")
        self._gui2_theme_label = ft.Text(size=12, data="footer_theme")
        # GUI2 requires an explicit confirmation before changing all settings
        # back to their defaults; the legacy GUI keeps its original behavior.
        self.reset_btn.on_click = self.confirm_reset_defaults
        # The legacy builder gives these controls a fixed width. GUI2 places
        # them in a compact footer, so let Flet size each box from its content.
        self._set_gui2_footer_dropdowns_adaptive()
        self._gui2_nav_expanded_width = self._calculate_gui2_nav_expanded_width()
        self._gui2_nav = self._build_navigation()
        self._gui2_nav.left = 0
        self._gui2_nav.top = 0
        self._gui2_nav.bottom = 0
        self._gui2_nav_stack = ft.Stack(
            controls=[self._gui2_nav],
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        self._gui2_nav_host = ft.Container(
            content=self._gui2_nav_stack,
            on_hover=self._on_gui2_navigation_hover,
            width=self._gui2_nav_expanded_width,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        self._set_gui2_navigation_layout(False)
        # GUI2 no longer renders a top menu bar. Keep the menu registry and
        # builders available for a future command surface without mounting it.
        self._gui2_menu = None
        self._page_host = ft.Container(expand=True)

        page_body = ft.Row(
            controls=[
                self._gui2_nav_host,
                ft.VerticalDivider(width=1),
                ft.Column(
                    controls=[
                        ft.Row([
                            self._gui2_page_title,
                            ft.Container(expand=True),
                            self._gui2_page_description,
                        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        ft.Divider(height=1),
                        self._page_host,
                    ],
                    expand=True,
                    spacing=8,
                ),
            ],
            expand=True,
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        footer = ft.Container(
            content=ft.Column([
                ft.Row([
                    self._gui2_language_label,
                    self.lang_dd,
                    self._gui2_theme_label,
                    self.theme_dd,
                    ft.Container(expand=True),
                    self.reset_btn,
                    self.stop_btn,
                    self.run_btn,
                ], spacing=8),
                self._backend_status_bar,
                ft.Container(
                    content=ft.Row([self._gui2_status], spacing=8),
                    padding=ft.Padding(10, 6, 10, 6),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
                    border_radius=6,
                ),
            ], spacing=5),
            padding=ft.Padding(0, 8, 0, 0),
        )
        self._gui2_shell = ft.Column(
            controls=[page_body, ft.Divider(height=1), footer],
            expand=True,
            spacing=0,
        )
        self._main_panel = self._gui2_shell
        self._root_row = self._gui2_shell
        self.page.add(self._gui2_shell)
        self._on_page_resize()
        self._refresh_gui2_texts()
        self._show_gui2_page(0, update=False)

    def _set_gui2_footer_dropdowns_adaptive(self):
        """Use intrinsic content width for GUI2's language and theme boxes."""
        for dropdown in (self.lang_dd, self.theme_dd):
            dropdown._fixed = None
            dropdown._min = 0
            dropdown._max = 0
            dropdown.reapply_width()

    def _section_page(self, title_key: str, rows: list[ft.Control], category: str) -> ft.Control:
        controls = [row for row in rows if row is not None]
        if not controls:
            controls = [ft.Text("No controls", size=12, color=ft.Colors.GREY)]
        return ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column(controls, spacing=8),
                    padding=ft.Padding(12, 12, 12, 12),
                    border=ft.Border.all(1, ft.Colors.OUTLINE),
                    border_radius=8,
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8,
            data=category,
        )

    def _build_quality_page(self) -> ft.Control:
        """Keep colour and projection-quality controls in separate groups."""
        groups = [
            ft.Container(
                content=ft.Column(self._gui2_quality_rows[:3], spacing=8),
                padding=ft.Padding(12, 12, 12, 12),
                border=ft.Border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
                data="quality_color_group",
            ),
            ft.Container(
                content=ft.Column(self._gui2_quality_rows[3:], spacing=8),
                padding=ft.Padding(12, 12, 12, 12),
                border=ft.Border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
                data="quality_lod_group",
            ),
        ]
        return ft.Column(
            controls=groups,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8,
            data="Image settings",
        )

    def _build_home_page(self, device_rows: list[ft.Control]) -> ft.Control:
        home_rows = [device_rows[index] for index in (11, 14, 16) if index < len(device_rows)]
        home_rows.append(self.stream_url_row)
        self._gui2_home_xr_rows = [
            device_rows[index] for index in (12, 13) if index < len(device_rows)
        ]
        self._gui2_home_streaming_rows = list(self._streamer_rows[1:])
        self._gui2_xr_container = ft.Container(
            content=ft.Column(self._gui2_home_xr_rows, spacing=8),
            padding=ft.Padding(16, 10, 16, 10),
            border=ft.Border.all(1, ft.Colors.OUTLINE),
            border_radius=6,
            data="home_xr_group",
        )
        self.stream_container.content.controls = [
            *self._gui2_home_streaming_rows,
        ]
        self._sync_gui2_home_streaming_visibility()
        return ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column([
                        *home_rows,
                    ], spacing=10),
                    padding=ft.Padding(16, 16, 16, 16),
                    border=ft.Border.all(1, ft.Colors.OUTLINE),
                    border_radius=8,
                ),
                self._gui2_xr_container,
                self.stream_container,
                ft.Container(
                    content=ft.Text(size=13, color=ft.Colors.GREY, data="home_tip"),
                    padding=ft.Padding(16, 12, 16, 12),
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8,
        )

    def _build_streaming_shortcut_page(self) -> ft.Control:
        """Keep the Streaming & XR navigation item as a Home shortcut."""
        return ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            data="streaming_shortcut_title",
                        ),
                        ft.Text(size=13, data="streaming_shortcut_body"),
                        ft.Button(
                            content=ft.Text(data="open_home_streaming"),
                            icon=ft.Icons.HOME_OUTLINED,
                            on_click=self.open_home_streaming,
                        ),
                    ], spacing=12),
                    padding=ft.Padding(16, 16, 16, 16),
                    border=ft.Border.all(1, ft.Colors.OUTLINE),
                    border_radius=8,
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8,
            data="Streaming & XR",
        )

    def _build_advanced_stereo_group(self) -> ft.Control:
        """Build the advanced stereo parameter area controlled by the checkbox."""
        group = ft.Container(
            content=ft.Column([
                *self._gui2_advanced_rows,
            ], spacing=8),
            padding=ft.Padding(0, 4, 0, 4),
            visible=bool(getattr(self.advanced_stereo_cb, "value", False)),
            data="advanced_stereo_group",
        )
        self._gui2_advanced_stereo_group = group
        return group

    def _build_advanced_shortcut_page(self) -> ft.Control:
        """Keep the Advanced navigation item as a discoverable shortcut."""
        return ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            data="advanced_shortcut_title",
                        ),
                        ft.Text(size=13, data="advanced_shortcut_body"),
                        ft.Button(
                            content=ft.Text(data="open_advanced_stereo"),
                            icon=ft.Icons.TUNE,
                            on_click=self.open_advanced_stereo,
                        ),
                    ], spacing=12),
                    padding=ft.Padding(16, 16, 16, 16),
                    border=ft.Border.all(1, ft.Colors.OUTLINE),
                    border_radius=8,
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8,
            data="Advanced",
        )

    def _build_navigation(self) -> ft.NavigationRail:
        return ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.NONE,
            extended=True,
            width=self._gui2_nav_expanded_width,
            min_width=72,
            min_extended_width=self._gui2_nav_expanded_width,
            scrollable=True,
            group_alignment=-1,
            use_indicator=True,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.HOME_OUTLINED, selected_icon=ft.Icons.HOME, label=""),
                ft.NavigationRailDestination(icon=ft.Icons.VIEW_IN_AR_OUTLINED, selected_icon=ft.Icons.VIEW_IN_AR, label=""),
                ft.NavigationRailDestination(icon=ft.Icons.PALETTE_OUTLINED, selected_icon=ft.Icons.PALETTE, label=""),
                ft.NavigationRailDestination(icon=ft.Icons.SPEED_OUTLINED, selected_icon=ft.Icons.SPEED, label=""),
                ft.NavigationRailDestination(icon=ft.Icons.TUNE_OUTLINED, selected_icon=ft.Icons.TUNE, label=""),
                ft.NavigationRailDestination(icon=ft.Icons.SUBJECT_OUTLINED, selected_icon=ft.Icons.SUBJECT, label=""),
                ft.NavigationRailDestination(icon=ft.Icons.HELP_OUTLINED, selected_icon=ft.Icons.HELP, label=""),
            ],
            on_change=self._on_gui2_navigation_change,
        )

    def _build_help_page(self) -> ft.Control:
        """Show project links, community material, and version in one panel."""
        qr_path = qr_asset_path()
        qr_control = (
            ft.Image(src=str(qr_path), width=240, height=240, fit=ft.BoxFit.CONTAIN)
            if qr_path is not None else
            ft.Text(size=13, color=ft.Colors.GREY, data="help_qr_missing")
        )
        group_number = QQ_GROUP_NUMBER or gui2_text(self.locale, "help_group_missing")
        return ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text(size=16, weight=ft.FontWeight.BOLD, data="help_panel_title"),
                        ft.Text(size=13, color=ft.Colors.GREY, data="help_panel_body"),
                        ft.Row([
                            ft.Button(
                                content=ft.Text(data="help_website"),
                                icon=ft.Icons.LANGUAGE,
                                on_click=self.open_website,
                            ),
                            ft.Text(WEBSITE_URL, selectable=True, size=12),
                        ], wrap=True),
                        ft.Divider(height=1),
                        ft.Text(size=14, weight=ft.FontWeight.BOLD, data="help_qq_title"),
                        qr_control,
                        ft.Row([
                            ft.Text(group_number, selectable=True, data="help_group_number"),
                            ft.Button(
                                content=ft.Text(data="help_copy_group"),
                                icon=ft.Icons.CONTENT_COPY,
                                disabled=not QQ_GROUP_NUMBER,
                                on_click=self.copy_qq_group_number,
                            ),
                            ft.Button(
                                content=ft.Text(data="help_open_invite"),
                                icon=ft.Icons.OPEN_IN_NEW,
                                disabled=not QQ_INVITE_URL,
                                on_click=self.open_qq_invite,
                            ),
                        ], wrap=True),
                        ft.Divider(height=1),
                        ft.Text(size=14, weight=ft.FontWeight.BOLD, data="help_about_title"),
                        ft.Text(size=13, data="help_about_body"),
                        ft.Text(f"Version {VERSION}", size=12, color=ft.Colors.GREY),
                    ], spacing=10),
                    padding=ft.Padding(16, 16, 16, 16),
                    border=ft.Border.all(1, ft.Colors.OUTLINE),
                    border_radius=8,
                    data="help_panel",
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8,
            data="Help",
        )

    def _on_gui2_navigation_hover(self, e):
        """Schedule delayed navigation expansion or collapse."""
        value = getattr(e, "data", False)
        pointer_inside = value is True or str(value).strip().lower() in {"true", "1"}
        if pointer_inside == self._gui2_nav_pointer_inside:
            return
        self._gui2_nav_pointer_inside = pointer_inside
        self._gui2_nav_hover_generation += 1
        generation = self._gui2_nav_hover_generation

        task = self._gui2_nav_hover_task
        if task is not None and not task.done():
            task.cancel()

        # Re-entering an already expanded rail only needs to cancel its pending
        # collapse. Likewise, leaving before expansion cancels the pending open.
        if pointer_inside == self._gui2_nav_hovered:
            self._gui2_nav_hover_task = None
            return

        loop = getattr(self, "_loop", None)
        if loop is None or loop.is_closed():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._gui2_nav_hover_task = None
                return
        delay = (
            GUI2_NAV_EXPAND_DELAY_SECONDS
            if pointer_inside
            else GUI2_NAV_COLLAPSE_DELAY_SECONDS
        )
        self._gui2_nav_hover_task = loop.create_task(
            self._apply_gui2_navigation_hover_after_delay(
                pointer_inside, delay, generation,
            )
        )

    async def _apply_gui2_navigation_hover_after_delay(
        self, expanded: bool, delay: float, generation: int,
    ):
        """Apply a hover state only when the pointer remained in that state."""
        try:
            await asyncio.sleep(delay)
            if (
                generation != self._gui2_nav_hover_generation
                or expanded != self._gui2_nav_pointer_inside
                or getattr(self, "_closed", False)
            ):
                return
            self._gui2_nav_hovered = expanded
            self._apply_gui2_navigation_hover_state(expanded)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            if self._gui2_nav_hover_task is current:
                self._gui2_nav_hover_task = None

    def _apply_gui2_navigation_hover_state(self, expanded: bool):
        """Commit the delayed hover state without reflowing rail contents."""
        # The rail itself stays extended so Flet keeps each label aligned with
        # its icon. Only the clipping host changes width, avoiding reflow.
        self._gui2_nav.extended = True
        self._gui2_nav.label_type = ft.NavigationRailLabelType.NONE
        self._set_gui2_navigation_layout(expanded)
        self.page.update()

    def _set_gui2_navigation_layout(self, expanded: bool):
        """Change only the clipped viewport width; never constrain the rail."""
        width = self._gui2_nav_expanded_width if expanded else GUI2_NAV_COLLAPSED_WIDTH
        previous_width = GUI2_NAV_COLLAPSED_WIDTH
        if self._gui2_nav_host is not None:
            previous_width = self._gui2_nav_host.width or GUI2_NAV_COLLAPSED_WIDTH
            self._gui2_nav_host.width = width
        if self._gui2_nav is not None:
            self._gui2_nav.width = self._gui2_nav_expanded_width
            self._gui2_nav.min_extended_width = self._gui2_nav_expanded_width
        # Preserve the right-hand content width. Without growing the native
        # window by the same delta, the expanded navigation only squeezes the
        # page instead of moving it as one block.
        width_delta = width - previous_width
        if width_delta and self._gui2_ready:
            current_window_width = getattr(self.page.window, "width", None) or 680
            self.page.window.width = max(680, current_window_width + width_delta)
            try:
                self.page.window.update()
            except RuntimeError:
                pass

    def _calculate_gui2_nav_expanded_width(self) -> int:
        """Size the expanded rail from the longest localized label."""
        labels = [gui2_text(self.locale, f"nav_{key}") for key in PAGE_KEYS]
        longest = max((
            sum(14 if ord(char) > 127 else 8 for char in label)
            for label in labels
        ), default=0)
        width = GUI2_NAV_COLLAPSED_WIDTH + longest + 32
        return max(
            GUI2_NAV_MIN_EXPANDED_WIDTH,
            min(GUI2_NAV_MAX_EXPANDED_WIDTH, width),
        )

    def _build_menu_bar(self) -> ft.MenuBar:
        return ft.MenuBar(
            controls=[self._menu_control(spec) for spec in build_menu_specs()],
            expand=True,
        )

    def _menu_control(self, spec: MenuItemSpec) -> ft.Control:
        if spec.children:
            return ft.SubmenuButton(
                content=ft.Text(gui2_text(self.locale, spec.label_key)),
                controls=[self._menu_control(child) for child in spec.children],
            )
        callback = getattr(self, spec.callback_name or "", None)
        return ft.MenuItemButton(
            content=ft.Text(self._menu_label(spec)),
            on_click=callback,
            disabled=not spec.enabled,
        )

    def _menu_label(self, spec: MenuItemSpec) -> str:
        label = gui2_text(self.locale, spec.label_key)
        if spec.item_id == f"language_{'cn' if self.locale == 'CN' else 'en'}":
            return f"✓ {label}"
        current_theme = self._current_theme_key()
        if spec.item_id == f"theme_{current_theme}":
            return f"✓ {label}"
        return label

    def _current_theme_key(self) -> str:
        value = str(getattr(self.theme_dd, "value", "system") or "system").lower()
        reverse = {"系统": "system", "蓝色": "blue", "绿色": "green", "红色": "red",
                   "紫色": "purple", "橙色": "orange", "青色": "teal", "粉色": "pink", "灰色": "grey"}
        return reverse.get(value, value)

    def _on_gui2_navigation_change(self, e):
        index = int(e.control.selected_index or 0)
        if PAGE_KEYS[index] == "advanced":
            self.open_advanced_stereo()
            return
        self._show_gui2_page(index)

    def _sync_gui2_home_streaming_visibility(self):
        """Keep the Home Streaming/XR section mounted and sync mode-only rows."""
        self.stream_container.visible = (
            self._is_streaming_run_mode()
            and bool(getattr(self.stream_settings_cb, "value", False))
        )
        for row in getattr(self, "_gui2_home_streaming_rows", self._streamer_rows[1:]):
            row.visible = True
        for row in getattr(self, "_gui2_home_xr_rows", []):
            row.visible = True
        for control_name in (
            "xr_headset_label", "xr_headset_dd", "display_mode_label",
            "display_mode_dd", "controller_label", "ctrl_model_dd",
            "environment_label", "env_model_dd",
        ):
            control = getattr(self, control_name, None)
            if control is not None:
                control.visible = True
        is_openxr = getattr(self, "run_mode_key", None) == "OpenXR Link"
        for control_name in (
            "controller_label", "ctrl_model_dd",
            "environment_label", "env_model_dd",
        ):
            control = getattr(self, control_name, None)
            if control is not None:
                self._set_gui2_disabled_visual(control, not is_openxr)
        # OpenXR obtains its presentation mode from the headset/runtime. Other
        # output modes continue to use the selectable GUI display mode.
        for control_name in ("display_mode_label", "display_mode_dd"):
            control = getattr(self, control_name, None)
            if control is not None:
                self._set_gui2_disabled_visual(control, is_openxr)
        self.stream_settings_cb.visible = self._is_streaming_run_mode()
        self.stream_url_row.visible = self._is_streaming_run_mode()

    @staticmethod
    def _set_gui2_disabled_visual(control, disabled: bool):
        """Disable a GUI2 control and visibly grey custom compact controls."""
        control.disabled = disabled
        control.opacity = 0.45 if disabled else 1.0
        # CompactDropdown wraps PopupMenuButton in a Container. Explicitly
        # disable that inner button because the custom label and border do not
        # receive Flet's native Dropdown disabled styling from the wrapper.
        content = getattr(control, "content", None)
        if isinstance(content, ft.PopupMenuButton):
            content.disabled = disabled

    def _is_streaming_run_mode(self) -> bool:
        return getattr(self, "run_mode_key", None) in {
            "MJPEG Streamer", "RTMP Streamer",
        }

    def _sync_advanced_stereo_visibility(self):
        """Extend the legacy advanced-stereo switch to GUI2-only rows."""
        super()._sync_advanced_stereo_visibility()
        visible = bool(
            getattr(self, "advanced_stereo_cb", None)
            and self.advanced_stereo_cb.value
        )
        for row in getattr(self, "_gui2_advanced_rows", []):
            row.visible = visible
        group = getattr(self, "_gui2_advanced_stereo_group", None)
        if group is not None:
            group.visible = visible
        # These rows are mounted on Performance in GUI2, so they are
        # independent of the Stereo advanced-parameter checkbox.
        for row in getattr(self, "_gui2_acceleration_rows", []):
            row.visible = True
        # The advanced rows materially change the Stereo page's natural
        # height. Resize the native window immediately after the checkbox is
        # toggled instead of retaining the height calculated for the previous
        # visibility state.
        if (
            getattr(self, "_gui2_ready", False)
            and group is not None
            and getattr(self, "_gui2_page_index", -1)
            == PAGE_KEYS.index("stereo")
        ):
            self._fit_window_to_content(update=True, resize_window=True)

    def _update_accelerator_visibility(self, device_label):
        """Keep every accelerator row visible and disable unsupported choices."""
        super()._update_accelerator_visibility(device_label)
        self._apply_gui2_acceleration_policy(device_label)

    def _apply_gui2_acceleration_policy(self, device_label=None):
        rows = getattr(self, "_gui2_acceleration_rows", [])
        if not rows:
            return
        device_label = device_label if device_label is not None else getattr(
            self.device_dd, "value", ""
        )
        device_label = device_label or ""
        cuda = "CUDA" in device_label
        rocm = cuda and devices_module.IS_ROCM
        supported = {
            "torch_compile_cb": cuda,
            "tensorrt_cb": cuda and not rocm,
            "recompile_trt_cb": cuda and not rocm,
            "coreml_cb": "MPS" in device_label,
            "recompile_coreml_cb": "MPS" in device_label,
            "openvino_cb": "XPU" in device_label,
            "recompile_openvino_cb": "XPU" in device_label,
            "migraphx_cb": rocm,
            "recompile_migraphx_cb": rocm,
        }
        for row in rows:
            row.visible = True
        for name, is_supported in supported.items():
            control = getattr(self, name, None)
            if control is not None:
                # GUI2 exposes the complete matrix; only platform support
                # determines whether a choice is enabled here.
                control.visible = True
                control.disabled = not is_supported
        if getattr(self, "acceleration_label", None) is not None:
            self.acceleration_label.visible = True

    def _sync_device_advanced_visibility(self, mode):
        """Keep GUI2 quality controls visible on their dedicated page."""
        super()._sync_device_advanced_visibility(mode)
        for row in getattr(self, "_gui2_quality_rows", []):
            row.visible = True
        self._apply_gui2_performance_visibility()

    def _apply_gui2_performance_visibility(self):
        """Keep all GUI2 Performance advanced rows and fields visible."""
        for row in getattr(self, "_gui2_performance_rows", []):
            row.visible = True
        for name in (
            "advanced_device_cb", "capture_tool_label", "capture_tool_dd",
            "target_fps_label", "target_fps_dd", "xr_preview_cb",
            "local_vsync_cb", "window_preview_cb", "render_scale_label",
            "render_scale_dd", "render_align_label", "render_align_dd",
        ):
            control = getattr(self, name, None)
            if control is not None:
                control.visible = True

    def open_advanced_stereo(self, _event=None):
        """Open Stereo and expand the advanced parameter group."""
        self.advanced_stereo_cb.value = True
        self._sync_advanced_stereo_visibility()
        self._show_gui2_page(PAGE_KEYS.index("stereo"))
        try:
            self.advanced_stereo_cb.update()
        except (AssertionError, RuntimeError):
            pass

    def _show_gui2_page(self, index: int, update: bool = True):
        index = max(0, min(index, len(PAGE_KEYS) - 1))
        self._gui2_page_index = index
        key = PAGE_KEYS[index]
        if not hasattr(self, "_page_host"):
            self._page_host = ft.Container(expand=True)
        self._page_host.content = self._gui2_pages[key]
        if self._gui2_nav is not None:
            self._gui2_nav.selected_index = index
        if self._gui2_page_title is not None:
            self._gui2_page_title.value = gui2_text(self.locale, f"{key}_title")
        if self._gui2_page_description is not None:
            self._gui2_page_description.value = gui2_text(
                self.locale, PAGE_DESCRIPTION_KEYS[key]
            )
        if self._gui2_status is not None:
            self._gui2_status.value = self.status_text.value or gui2_text(self.locale, "status_idle")
        if update:
            # Commit the new page first, then resize its native window. This
            # avoids fitting against the previous page's client-side layout.
            self.page.update()
            self._fit_window_to_content(update=False, resize_window=True)

    def _refresh_gui2_texts(self):
        if self._gui2_menu is not None:
            self._gui2_menu.controls = [self._menu_control(spec) for spec in build_menu_specs()]
        if self._gui2_nav is not None:
            labels = [gui2_text(self.locale, f"nav_{key}") for key in PAGE_KEYS]
            for destination, label in zip(self._gui2_nav.destinations, labels):
                destination.label = label
            self._gui2_nav_expanded_width = self._calculate_gui2_nav_expanded_width()
            self._set_gui2_navigation_layout(self._gui2_nav_hovered)
        for page in self._gui2_pages.values():
            for control in self._walk_controls(page):
                key = getattr(control, "data", None)
                if isinstance(key, str) and key in (
                    "home_core", "home_description", "home_tip",
                    "home_streaming_title", "streaming_shortcut_title",
                    "streaming_shortcut_body", "open_home_streaming",
                    "advanced_group_hint", "advanced_shortcut_title",
                    "advanced_shortcut_body", "open_advanced_stereo",
                    "footer_language", "footer_theme",
                    "help_title", "help_panel_title", "help_panel_body",
                    "help_qr_missing", "help_qq_title", "help_website",
                    "help_group_missing", "help_copy_group", "help_open_invite",
                    "help_about_title", "help_about_body",
                ):
                    control.value = gui2_text(self.locale, key)
        help_group_number = getattr(self, "_gui2_pages", {}).get("help")
        if help_group_number is not None:
            for control in self._walk_controls(help_group_number):
                if getattr(control, "data", None) == "help_group_number":
                    control.value = QQ_GROUP_NUMBER or gui2_text(self.locale, "help_group_missing")
        self._show_gui2_page(self._gui2_page_index, update=False)

    @staticmethod
    def _walk_controls(control):
        yield control
        for child in getattr(control, "controls", []) or []:
            yield from Desktop2StereoGUI2._walk_controls(child)
        content = getattr(control, "content", None)
        if content is not None:
            yield from Desktop2StereoGUI2._walk_controls(content)

    def _invoke_language(self, locale: str, _event=None):
        self.lang_dd.value = "简体中文" if locale == "CN" else "English"
        super().on_language_change(SimpleNamespace(control=self.lang_dd))
        self._refresh_gui2_texts()
        self.page.update()

    def select_language_en(self, e=None):
        self._invoke_language("EN", e)

    def select_language_cn(self, e=None):
        self._invoke_language("CN", e)

    def _invoke_theme(self, theme: str, _event=None):
        self.theme_dd.value = gui2_text(self.locale, f"theme_{theme}")
        super().on_theme_change(SimpleNamespace(control=self.theme_dd))
        self._refresh_gui2_texts()
        self.page.update()

    def __getattr__(self, name):
        if name.startswith("select_theme_"):
            return lambda e=None: self._invoke_theme(name.removeprefix("select_theme_"), e)
        raise AttributeError(name)

    def open_diagnostics(self, _event=None):
        self._show_gui2_page(PAGE_KEYS.index("logs"))

    def open_website(self, _event=None):
        try:
            self.page.launch_url(WEBSITE_URL)
        except Exception as exc:
            self._show_gui2_snackbar(gui2_text(self.locale, "website_open_failed", exc))

    def open_qq_group(self, _event=None):
        qr_path = qr_asset_path()
        content = [
            ft.Text(gui2_text(self.locale, "qq_missing_qr") if qr_path is None else ""),
        ]
        if qr_path is not None:
            content = [ft.Image(src=str(qr_path), width=240, height=240, fit=ft.BoxFit.CONTAIN)]
        group_text = QQ_GROUP_NUMBER or gui2_text(self.locale, "qq_missing_number")
        content.extend([
            ft.Text(group_text, selectable=True),
            ft.Row([
                ft.Button(content=ft.Text(gui2_text(self.locale, "copy_group_number")),
                          disabled=not QQ_GROUP_NUMBER, on_click=self.copy_qq_group_number),
                ft.Button(content=ft.Text(gui2_text(self.locale, "open_invite")),
                          disabled=not QQ_INVITE_URL, on_click=self.open_qq_invite),
            ], wrap=True),
        ])
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(gui2_text(self.locale, "qq_title")),
            content=ft.Column(content, width=360, spacing=12),
            actions=[ft.Button(content=ft.Text(gui2_text(self.locale, "close")), on_click=self._close_gui2_dialog)],
        )
        self.page.show_dialog(dialog)

    async def copy_qq_group_number(self, _event=None):
        if not QQ_GROUP_NUMBER:
            self._show_gui2_snackbar(gui2_text(self.locale, "group_number_missing"))
            return
        try:
            await ft.Clipboard().set(QQ_GROUP_NUMBER)
            self._show_gui2_snackbar(gui2_text(self.locale, "group_number_copied"))
        except Exception as exc:
            self._show_gui2_snackbar(gui2_text(self.locale, "group_number_missing") + f" ({exc})")

    def open_qq_invite(self, _event=None):
        if not QQ_INVITE_URL:
            self._show_gui2_snackbar(gui2_text(self.locale, "invite_open_failed", "not configured"))
            return
        try:
            self.page.launch_url(QQ_INVITE_URL)
        except Exception as exc:
            self._show_gui2_snackbar(gui2_text(self.locale, "invite_open_failed", exc))

    def open_about(self, _event=None):
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(gui2_text(self.locale, "about_title")),
            content=ft.Column([
                ft.Text(gui2_text(self.locale, "about_body")),
                ft.Text(f"Version {VERSION}"),
                ft.Text(WEBSITE_URL, selectable=True),
            ], width=360, spacing=10),
            actions=[ft.Button(content=ft.Text(gui2_text(self.locale, "close")), on_click=self._close_gui2_dialog)],
        )
        self.page.show_dialog(dialog)

    def confirm_reset_defaults(self, _event=None):
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(gui2_text(self.locale, "reset_confirm_title")),
            content=ft.Text(gui2_text(self.locale, "reset_confirm_body")),
            actions=[
                ft.Button(content=ft.Text(gui2_text(self.locale, "cancel")), on_click=self._close_gui2_dialog),
                ft.Button(content=ft.Text(gui2_text(self.locale, "confirm")), on_click=self._confirm_reset_defaults),
            ],
        )
        self.page.show_dialog(dialog)

    def _confirm_reset_defaults(self, _event=None):
        self._close_gui2_dialog()
        self.reset_defaults(_event)

    def _close_gui2_dialog(self, _event=None):
        try:
            self.page.pop_dialog()
        except Exception:
            pass

    def _show_gui2_snackbar(self, message: str):
        try:
            self.page.show_dialog(ft.SnackBar(ft.Text(message)))
        except Exception:
            self.set_status(message)

    def on_language_change(self, e):
        # The GUI2 Chinese label for the system theme is intentionally
        # shortened to “主题”; normalize it before the legacy handler maps
        # display text to the stable internal ``system`` value.
        restore_theme_label = False
        if getattr(self, "theme_dd", None) is not None and self.theme_dd.value == "主题":
            self.theme_dd.value = "系统"
            restore_theme_label = True
        super().on_language_change(e)
        if restore_theme_label and self.locale == "CN":
            self.theme_dd.value = gui2_text(self.locale, "theme_system")
        self._refresh_gui2_texts()

    def on_theme_change(self, e):
        restore_theme_label = False
        if self.locale == "CN" and getattr(e.control, "value", None) == "主题":
            e.control.value = "系统"
            restore_theme_label = True
        super().on_theme_change(e)
        if restore_theme_label:
            e.control.value = gui2_text(self.locale, "theme_system")
        self._refresh_gui2_texts()

    def update_ui_texts(self):
        super().update_ui_texts()
        self._refresh_gui2_texts()

    def _set_log_panel_visible(self, visible, save=False, update=True):
        self._gui2_log_requested = bool(visible)
        if visible and self._gui2_pages and self._gui2_ready:
            self._show_gui2_page(PAGE_KEYS.index("logs"), update=update)
        elif update and hasattr(self, "page"):
            self.page.update()

    def on_log_visibility_link(self, _event=None):
        self.open_diagnostics()

    def _show_streamer_rows(self, *row_indices):
        """Keep every streamer row mounted and visible in GUI2."""
        col = self.stream_container.content.controls
        col.clear()
        col.extend(getattr(self, "_gui2_home_streaming_rows", self._streamer_rows[1:]))
        self._sync_gui2_home_streaming_visibility()
        try:
            mounted = self.stream_container.page is not None
        except RuntimeError:
            mounted = False
        if mounted:
            self.stream_container.update()
        self._fit_window_to_content()

    def _fit_window_to_content(self, update=True, resize_window=False):
        # Each GUI2 page has a different control matrix. Recalculate both
        # native dimensions from that page instead of retaining the previous
        # page's geometry or counting controls inside hidden containers.
        self.page.window.min_width = 680
        self.page.window.min_height = 480
        if resize_window:
            width, height = self._estimate_gui2_window_size()
            self.page.window.width = width
            self.page.window.height = height
            try:
                self.page.window.update()
            except RuntimeError:
                pass
        if update:
            self.page.update()

    def _estimate_gui2_window_size(self) -> tuple[int, int]:
        """Estimate native dimensions from the currently visible page only."""
        page = self._gui2_pages.get(PAGE_KEYS[self._gui2_page_index])
        if page is None:
            return 960, 640
        visible_rows = 0
        visible_groups = 0
        has_large_visual = False
        max_content_width = 0

        def visit(control, parent_visible=True):
            nonlocal visible_rows, visible_groups, has_large_visual, max_content_width
            visible = parent_visible and getattr(control, "visible", True)
            if not visible:
                return 0
            if isinstance(control, ft.Row) and getattr(control, "controls", None):
                visible_rows += 1
                # A GUI parameter row contains internal Rows inside compact
                # dropdowns and buttons. Treat the semantic row as one unit
                # and do not count those implementation details again.
                width = self._estimate_gui2_control_width(control)
                max_content_width = max(max_content_width, width)
                return width
            if (
                isinstance(control, ft.Container)
                and getattr(control, "border", None) is not None
            ):
                visible_groups += 1
            if isinstance(control, ft.Image):
                has_large_visual = True
            width = self._estimate_gui2_control_width(control)
            max_content_width = max(max_content_width, width)
            children = list(getattr(control, "controls", []) or [])
            content = getattr(control, "content", None)
            if content is not None:
                children.append(content)
            child_width = max((visit(child, visible) for child in children), default=0)
            return max(width, child_width)

        content_width = visit(page)
        max_content_width = max(max_content_width, content_width)
        # Navigation rail, divider, page margins, and a small safety margin.
        nav_extra = (
            self._gui2_nav_expanded_width - GUI2_NAV_COLLAPSED_WIDTH
            if self._gui2_nav_hovered else 0
        )
        width = max(680, min(1600, max_content_width + 120 + nav_extra))
        # Compact GUI2 parameter controls occupy about 30 px plus 8 px row
        # spacing. The old 46 px/row and 290 px shell allowance compounded on
        # dense pages and left a large empty strip above the fixed footer.
        # Keep the estimate tied to this page's visible semantic rows while
        # retaining a small allowance for its header, footer, and margins.
        estimate = 220 + visible_rows * 38 + visible_groups * 20
        if has_large_visual:
            estimate += 250
        return width, max(480, min(1000, estimate))

    def _estimate_gui2_control_width(self, control) -> int:
        """Estimate a control's natural width without requiring client layout."""
        explicit = getattr(control, "width", None)
        if isinstance(explicit, (int, float)) and explicit > 0:
            return int(explicit)
        if isinstance(control, ft.Row):
            children = list(getattr(control, "controls", []) or [])
            spacing = getattr(control, "spacing", 0) or 0
            return sum(self._estimate_gui2_control_width(child) for child in children) + max(0, len(children) - 1) * spacing
        if isinstance(control, ft.Column):
            return max((self._estimate_gui2_control_width(child) for child in control.controls), default=0)
        if isinstance(control, ft.Image):
            return int(getattr(control, "width", None) or 240)
        value = getattr(control, "value", None) or getattr(control, "label", None)
        if value:
            return min(900, max(48, sum(14 if ord(char) > 127 else 8 for char in str(value)) + 28))
        return 48

    # The legacy startup-fit task calls these two hooks directly. Override
    # them so its delayed recalculation cannot replace GUI2's page-specific
    # dimensions with the legacy single-page estimate.
    def _estimate_window_width(self, main_width=None):
        return self._estimate_gui2_window_size()[0]

    def _estimate_window_height(self):
        return self._estimate_gui2_window_size()[1]

    def _on_page_resize(self, e=None):
        width = getattr(getattr(self.page, "window", None), "width", None) or 960
        compact = width < 1040
        if compact != self._gui2_compact and self._gui2_nav is not None:
            self._gui2_compact = compact
            self._set_gui2_navigation_layout(self._gui2_nav_hovered)
            self.page.update()

    def set_status(self, message, key=None):
        super().set_status(message, key=key)
        if self._gui2_status is not None:
            self._gui2_status.value = message


async def _async_main(page: ft.Page):
    page.window.width = 960
    page.window.height = 640
    page.padding = 16
    page.add(ft.ProgressRing(width=28, height=28))
    page.update()
    app = Desktop2StereoGUI2(page)
    await app.setup()


def main():
    ft.run(_async_main)
