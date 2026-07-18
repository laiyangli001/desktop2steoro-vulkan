import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


display_module = _load_module("display_module", Path("src") / "utils" / "display.py")
preprocess_module = _load_module("preprocess_module", Path("src") / "capture" / "preprocess.py")
compute_output_resolution = display_module.compute_output_resolution
capture_frame_to_rgb = preprocess_module.capture_frame_to_rgb


def test_full_sbs_wxh_resolution_uses_half_target_width_per_eye():
    assert compute_output_resolution("7680x4320", "Full-SBS", 1, 2) == (3840, 4320)
    assert compute_output_resolution("3840x2160", "Full-SBS", 1, 2) == (1920, 2160)


def test_half_sbs_wxh_resolution_keeps_target_canvas_size():
    assert compute_output_resolution("3840x2160", "Half-SBS", 1, 2) == (3840, 2160)


def test_full_tab_wxh_resolution_uses_half_target_height_per_eye():
    assert compute_output_resolution("3840x2160", "Full-TAB", 1, 2) == (3840, 1080)


def test_numeric_height_resolution_keeps_legacy_meaning():
    assert compute_output_resolution("2160", "Full-SBS", 1, 2) == 2160


def test_wxh_resolution_accepts_common_separators():
    assert compute_output_resolution("7680 * 4320", "full_sbs", 1, 2) == (3840, 4320)
    assert compute_output_resolution("7680x4320", "full_sbs", 1, 2) == (3840, 4320)


def test_auto_3d_monitor_fits_4k_input_to_1080p_output(monkeypatch):
    sizes = {1: (3840, 2160), 2: (1920, 1080)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (1920, 1080)


def test_auto_3d_monitor_keeps_1080p_input_for_4k_output(monkeypatch):
    sizes = {1: (1920, 1080), 2: (3840, 2160)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (1920, 1080)


def test_auto_3d_monitor_keeps_4k_input_for_4k_output(monkeypatch):
    sizes = {1: (3840, 2160), 2: (3840, 2160)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (3840, 2160)


def test_auto_3d_monitor_fits_4k_input_to_2k_output(monkeypatch):
    sizes = {1: (3840, 2160), 2: (2560, 1440)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (2560, 1440)


def test_auto_3d_monitor_keeps_2k_input_for_4k_output(monkeypatch):
    sizes = {1: (2560, 1440), 2: (3840, 2160)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (2560, 1440)


def test_auto_3d_monitor_preserves_input_aspect_for_ultrawide_output(monkeypatch):
    sizes = {1: (3840, 2160), 2: (3440, 1440)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (2560, 1440)


def test_auto_3d_monitor_preserves_input_aspect_for_16_10_output(monkeypatch):
    sizes = {1: (3840, 2160), 2: (2560, 1600)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (2560, 1440)


def test_auto_3d_monitor_keeps_21_9_input_when_output_is_larger(monkeypatch):
    sizes = {1: (3440, 1440), 2: (5120, 2160)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (3440, 1440)


def test_auto_3d_monitor_fits_21_9_input_to_16_9_output_without_stretch(monkeypatch):
    sizes = {1: (3440, 1440), 2: (2560, 1440)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (2560, 1072)


def test_auto_3d_monitor_keeps_32_9_input_when_output_is_larger(monkeypatch):
    sizes = {1: (5120, 1440), 2: (7680, 2160)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (5120, 1440)


def test_auto_3d_monitor_fits_32_9_input_to_16_9_output_without_stretch(monkeypatch):
    sizes = {1: (5120, 1440), 2: (3840, 2160)}
    monkeypatch.setattr(display_module, "get_monitor_size", lambda idx=None: sizes[idx])

    assert compute_output_resolution("Auto", "Half-SBS", 1, 2, use_stereo_monitor=True) == (3840, 1080)


def test_capture_preprocess_accepts_exact_width_height_target():
    frame_bgra = np.zeros((4, 8, 4), dtype=np.uint8)
    frame_bgra[..., 0] = 10
    frame_bgra[..., 1] = 20
    frame_bgra[..., 2] = 30
    frame_bgra[..., 3] = 255

    frame_rgb = capture_frame_to_rgb(frame_bgra, (6, 10))

    assert frame_rgb.shape == (10, 6, 3)
    assert tuple(frame_rgb[0, 0]) == (30, 20, 10)
