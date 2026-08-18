from __future__ import annotations

from utils.display_info import (
    DisplayInfo,
    _windows_instance_key,
    match_display_index,
    resolve_glfw_monitor_index,
)


class _Size:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


class _Mode:
    def __init__(self, width: int, height: int) -> None:
        self.size = _Size(width, height)


class _Glfw:
    monitors = ["primary", "second", "third"]
    geometry = {
        "primary": ((0, 0), (3840, 2160)),
        "second": ((3840, 0), (1920, 1080)),
        "third": ((5760, 0), (3840, 2160)),
    }

    @classmethod
    def get_monitors(cls):
        return cls.monitors

    @classmethod
    def get_monitor_pos(cls, monitor):
        return cls.geometry[monitor][0]

    @classmethod
    def get_video_mode(cls, monitor):
        return _Mode(*cls.geometry[monitor][1])

    @staticmethod
    def get_monitor_name(_monitor):
        return b"Generic PnP Monitor"


DISPLAYS = [
    DisplayInfo(3, 1, 0, 0, 3840, 2160, True, "hp-id", "HP Z27s", model="HP Z27s"),
    DisplayInfo(1, 2, 3840, 0, 1920, 1080, False, "dell-id", "DELL E2215HV", model="DELL E2215HV"),
    DisplayInfo(2, 3, 5760, 0, 3840, 2160, False, "viture-id", "VITURE", model="VITURE"),
]


def test_glfw_mapping_uses_display_identity_instead_of_mss_order() -> None:
    assert resolve_glfw_monitor_index(2, _Glfw, DISPLAYS) == 2
    assert resolve_glfw_monitor_index(1, _Glfw, DISPLAYS) == 1
    assert resolve_glfw_monitor_index(3, _Glfw, DISPLAYS) == 0


def test_display_matching_prefers_model_and_serial_for_duplicate_models() -> None:
    target = DisplayInfo(2, 2, 1920, 0, 1920, 1080, model="VITURE", serial="B")
    candidates = [
        DisplayInfo(1, 1, 0, 0, 1920, 1080, model="VITURE", serial="A"),
        DisplayInfo(2, 2, 1920, 0, 1920, 1080, model="VITURE", serial="B"),
    ]

    assert match_display_index(target, candidates) == 1


def test_display_matching_falls_back_to_geometry_when_names_are_generic() -> None:
    target = DisplayInfo(2, 3, 5760, 0, 3840, 2160, name="Generic PnP Monitor")
    candidates = [
        DisplayInfo(1, 1, 0, 0, 3840, 2160, name="Generic PnP Monitor"),
        DisplayInfo(2, 2, 5760, 0, 3840, 2160, name="Generic PnP Monitor"),
    ]

    assert match_display_index(target, candidates) == 1


def test_windows_mss_and_wmi_ids_normalize_to_the_same_instance() -> None:
    mss_id = r"\\?\DISPLAY#MTT1337#1&33320f0&0&UID256#{device-interface-guid}"
    wmi_id = r"DISPLAY\MTT1337\1&33320f0&0&UID256_0"

    assert _windows_instance_key(mss_id) == _windows_instance_key(wmi_id)
