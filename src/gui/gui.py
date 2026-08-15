"""Desktop2Stereo Flet GUI — main application class combining all mixins.

Mixins:
  GUIBuilderMixin  — UI construction, layout sizing, data population
  GUIHandlerMixin  — event handlers, visibility sync, i18n, audio, refresh
  GUIConfigMixin   — config read/write, stereo preset data, hot-param save
  GUIProcessMixin  — subprocess lifecycle, ESC monitoring, URL actions
"""
import os
import asyncio
import logging
from .flet_runtime import ensure_vendored_flet_view

ensure_vendored_flet_view()

import flet as ft
from utils import VERSION, OS_NAME, bootstrap_settings
from .builders import GUIBuilderMixin
from .handlers import GUIHandlerMixin
from .config_mgr import GUIConfigMixin
from .process import GUIProcessMixin, _setup_console_logging
from .config import DEFAULTS
from .controls import S
from .paths import BASE_DIR, GUI_READY_FILE, LOG_DIR
from .localization import UI_MESSAGES


logger = logging.getLogger(__name__)


class Desktop2StereoGUI(
    GUIBuilderMixin,
    GUIHandlerMixin,
    GUIConfigMixin,
    GUIProcessMixin,
):
    """Flet GUI for Desktop2Stereo — full equivalent of tk ConfigGUI."""
    def __init__(self, page: ft.Page):
        self.page = page
        self._loop = None
        self.locale = "EN"
        self._config = {}
        self.run_mode_key = DEFAULTS.get("Run Mode", "Local Viewer")
        self.capture_mode_key = DEFAULTS.get("Capture Mode", "Monitor")
        self.stream_protocol_key = DEFAULTS.get("Stream Protocol", "RTMP")
        self.selected_window_name = ""
        self.selected_window_handle = None
        self.selected_window_rect = None
        self._window_objects = []
        self.process = None
        self._starting = False
        self._proc_lock = None
        self.monitor_label_to_index = {}
        self.device_label_to_index = {}
        self._esc_down = None
        self._esc_stopped = False
        self._closed = False
        self._cancel_starting = False
        self._stopping = False
        self._labels_aligned = False
        self._status_key = ""
        self._local_ip_cache = "127.0.0.1"
        self._local_ip_task = None
        self.gui_log_handler = None
        self._log_poll_task = None
        self._resize_repaint_task = None
        self._media_change_task = None
        self._monitor_settle_task = None
        self._content_measure_task = None
        self._startup_layout_task = None
        self._pending_page_width = None
        self._pending_page_height = None
        self._ui_layout_lock = False
        self._ui_programmatic_resize = False
        self._ui_programmatic_resize_until = 0.0
        self._ui_monitor_transition_until = 0.0
        self._ui_last_window_width = 0.0
        self._ui_last_window_height = 0.0
        self._ui_base_page_padding = S(24)
        self._ui_base_viewport_width = 0.0
        self._ui_base_root_width = 0.0
        self._ui_base_root_height = 0.0
        self._ui_settings_content_height = 0.0
        self.display_scale = 1.0
        self.viewport_scale = 1.0
        self.effective_ui_scale = 1.0
        self.device_pixel_ratio = 1.0
        self._progress_log_spans = {}

    async def setup(self):
        self.gui_log_handler = _setup_console_logging()
        self._loop = asyncio.get_running_loop()
        self._proc_lock = asyncio.Lock()
        self._hot_save_task = None

        media = getattr(self.page, "media", None)
        self.device_pixel_ratio = float(
            getattr(media, "device_pixel_ratio", 1.0) or 1.0
        )
        self._refresh_display_scale()
        self.viewport_scale = 1.0
        self.effective_ui_scale = self.display_scale

        self.page.title = f"Desktop2Stereo v{VERSION}"
        self.page.window.icon = os.path.join(BASE_DIR, "icon.ico")
        self.page.padding = self._ui_base_page_padding * self.effective_ui_scale
        self.page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        if OS_NAME == "Windows":
            font = "Microsoft YaHei"
        elif OS_NAME == "Darwin":
            font = "PingFang SC"
        else:
            font = "Noto Sans SC"
        self.page.theme = ft.Theme(color_scheme_seed="blue", font_family=font)
        self.page.spacing = 0
        self.page.theme_mode = ft.ThemeMode.SYSTEM
        self.page.window.min_width = S(520) * 0.5
        self.page.window.min_height = S(300) * 0.5

        # Build UI
        self.build_ui()
        self._log_poll_task = asyncio.create_task(self._poll_log_queue())
        self._auto_align_labels()
        self.page.on_close = self._on_page_close
        self.page.on_resize = self._on_page_resize
        self.page.on_media_change = self._on_page_media_change
        self.page.window.on_event = self._on_window_event

        # Populate monitors & devices
        self.monitor_label_to_index = self.populate_monitors()
        self.device_label_to_index = self.populate_devices()

        # Load config
        self._config = DEFAULTS.copy()
        if os.path.exists(os.path.join(BASE_DIR, "settings.yaml")):
            try:
                cfg = bootstrap_settings(os.path.join(BASE_DIR, "settings.yaml"), os_name=OS_NAME)
                if cfg:
                    self._config.update(cfg)
                    self._yaml_loaded = True
                    self.locale = self._config.get("Language", "EN")
                    os.environ["DESKTOP2STEREO_LOCALE"] = self.locale
                    self.apply_config(self._config)
                    self.set_status(UI_MESSAGES[self.locale]["Loaded settings.yaml at startup"],
                                    key="Loaded settings.yaml at startup")
            except Exception as e:
                self.apply_config(self._config)
                self.set_status(
                    f"{UI_MESSAGES[self.locale]['Failed to load settings.yaml:']} {e}")
        else:
            self.apply_config(self._config)

        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self.page.on_keyboard_event = self._on_key
        self._esc_task = asyncio.ensure_future(self._esc_poll_task())
        self._set_log_panel_visible(self._config.get("Show Log Panel", DEFAULTS["Show Log Panel"]), update=False)
        self._fit_window_to_content(update=False)
        self.page.window.visible = True
        self.page.update()
        await asyncio.sleep(0)
        self._fit_window_to_content(update=True, resize_window=True)
        self._signal_gui_ready()
        self._startup_layout_task = asyncio.create_task(
            self._stabilize_startup_layout()
        )
        asyncio.create_task(self._prepare_startup_after_window_visible())

    def _signal_gui_ready(self):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(GUI_READY_FILE, "w", encoding="utf-8") as ready_file:
                ready_file.write("ready\n")
        except Exception:
            logger.exception("Failed to write GUI ready flag")

    async def _stabilize_startup_layout(self, delays=(0.35,)):
        """Refit after Windows/Flet has applied its initial DPI client-area change."""
        try:
            for delay in delays:
                await asyncio.sleep(delay)
                if self._closed:
                    return
                media = getattr(self.page, "media", None)
                self.device_pixel_ratio = float(
                    getattr(media, "device_pixel_ratio", self.device_pixel_ratio)
                    or self.device_pixel_ratio
                )
                # Match a normal control-triggered refit.  The resize/media
                # handlers have already established the final viewport scale;
                # resetting it here causes a visible second layout jump.
                self._fit_window_to_content(update=True, resize_window=True)
        except asyncio.CancelledError:
            return
        except RuntimeError:
            return

    async def _prepare_startup_after_window_visible(self):
        self.set_status(
            UI_MESSAGES[self.locale].get(
                "Preparing Flet package...",
                "Preparing Flet desktop client...",
            ),
            key="Preparing Flet package...",
        )
        try:
            await asyncio.to_thread(ensure_vendored_flet_view)
            self.set_status(
                UI_MESSAGES[self.locale].get(
                    "Startup preparation complete",
                    "Startup preparation complete.",
                ),
                key="Startup preparation complete",
            )
        except Exception as exc:
            logger.exception("Startup preparation failed")
            message = UI_MESSAGES[self.locale].get(
                "Startup preparation failed: {}",
                "Startup preparation failed: {}",
            )
            self.set_status(message.format(exc))


def main():
    """Entry point for the GUI application."""
    _setup_console_logging()
    ft.run(_async_main)


async def _async_main(page: ft.Page):
    app = Desktop2StereoGUI(page)
    await app.setup()


if __name__ == "__main__":
    main()
