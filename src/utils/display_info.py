from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
import re
import subprocess
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DisplayInfo:
    capture_index: int
    display_number: int
    left: int
    top: int
    width: int
    height: int
    is_primary: bool = False
    stable_id: str | None = None
    name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial: str | None = None
    device_display_number: int | None = None

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def label_name(self) -> str | None:
        return self.model or self.name

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def enumerate_displays(os_name: str | None = None) -> list[DisplayInfo]:
    platform_name = os_name or _platform_name()
    displays = _mss_displays()
    if not displays:
        return []
    if platform_name == "Windows":
        displays = _enrich_windows(displays)
    elif platform_name == "Darwin":
        displays = _enrich_macos(displays)
    else:
        displays = _enrich_linux(displays)
    return _assign_display_numbers(displays, platform_name)


def resolve_glfw_monitor_index(
    capture_index: int,
    glfw: Any,
    displays: Iterable[DisplayInfo] | None = None,
) -> int:
    glfw_monitors = list(glfw.get_monitors() or [])
    if not glfw_monitors:
        return 0
    known = list(displays) if displays is not None else enumerate_displays()
    target = next((item for item in known if item.capture_index == int(capture_index)), None)
    if target is None:
        return _clamp_monitor_index(capture_index, len(glfw_monitors))

    candidates: list[DisplayInfo] = []
    for index, monitor in enumerate(glfw_monitors, start=1):
        left, top = glfw.get_monitor_pos(monitor)
        mode = glfw.get_video_mode(monitor)
        width, height = int(mode.size.width), int(mode.size.height)
        mapped = _display_by_geometry(known, int(left), int(top), width, height)
        if mapped is not None:
            candidates.append(mapped)
            continue
        raw_name = glfw.get_monitor_name(monitor)
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="replace")
        candidates.append(
            DisplayInfo(
                capture_index=index,
                display_number=index,
                left=int(left),
                top=int(top),
                width=width,
                height=height,
                name=str(raw_name or "").strip() or None,
            )
        )

    matched = match_display_index(target, candidates)
    if matched is not None:
        return matched
    return _clamp_monitor_index(capture_index, len(glfw_monitors))


def match_display_index(target: DisplayInfo, candidates: Iterable[DisplayInfo]) -> int | None:
    candidate_list = list(candidates)
    stable_id = _normalized_text(target.stable_id)
    if stable_id:
        for index, candidate in enumerate(candidate_list):
            if _normalized_text(candidate.stable_id) == stable_id:
                return index

    target_model = _normalized_text(target.model or target.name)
    target_serial = _normalized_text(target.serial)
    if target_model:
        same_model = [
            (index, candidate)
            for index, candidate in enumerate(candidate_list)
            if _normalized_text(candidate.model or candidate.name) == target_model
        ]
        if target_serial:
            for index, candidate in same_model:
                if _normalized_text(candidate.serial) == target_serial:
                    return index
        if len(same_model) == 1:
            return same_model[0][0]
        for index, candidate in same_model:
            if candidate.rect == target.rect:
                return index
        for index, candidate in same_model:
            if candidate.display_number == target.display_number:
                return index

    for index, candidate in enumerate(candidate_list):
        if candidate.rect == target.rect:
            return index
    return None


def _mss_displays() -> list[DisplayInfo]:
    try:
        import mss

        with mss.mss() as capture:
            monitors = list(capture.monitors[1:])
    except Exception:
        return []
    return [
        DisplayInfo(
            capture_index=index,
            display_number=index,
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
            is_primary=bool(monitor.get("is_primary", False)),
            stable_id=str(monitor.get("unique_id") or "").strip() or None,
            name=str(monitor.get("name") or "").strip() or None,
        )
        for index, monitor in enumerate(monitors, start=1)
    ]


def _enrich_windows(displays: list[DisplayInfo]) -> list[DisplayInfo]:
    metadata = _windows_wmi_metadata()
    device_numbers: dict[tuple[int, int, int, int], int] = {}
    try:
        import win32api

        for handle, _hdc, rect in win32api.EnumDisplayMonitors():
            info = win32api.GetMonitorInfo(handle)
            number = _windows_display_number(info.get("Device", ""))
            if number is not None:
                device_numbers[tuple(int(value) for value in rect)] = number
    except Exception:
        pass

    enriched = []
    for display in displays:
        monitor_metadata = metadata.get(_windows_instance_key(display.stable_id), {})
        enriched.append(
            replace(
                display,
                stable_id=monitor_metadata.get("stable_id") or display.stable_id,
                name=monitor_metadata.get("model") or display.name,
                manufacturer=monitor_metadata.get("manufacturer"),
                model=monitor_metadata.get("model"),
                serial=monitor_metadata.get("serial"),
                device_display_number=device_numbers.get(display.rect),
            )
        )
    return enriched


def _windows_wmi_metadata() -> dict[str, dict[str, str | None]]:
    try:
        import pythoncom
        import win32com.client
    except Exception:
        return {}
    pythoncom.CoInitialize()
    service = None
    rows = None
    row = None
    try:
        service = win32com.client.GetObject(r"winmgmts:\\.\root\wmi")
        rows = service.ExecQuery("SELECT * FROM WmiMonitorID")
        result = {}
        for row in rows:
            stable_id = str(row.InstanceName).removesuffix("_0")
            result[_windows_instance_key(stable_id)] = {
                "stable_id": stable_id,
                "manufacturer": _decode_wmi_chars(row.ManufacturerName),
                "model": _decode_wmi_chars(row.UserFriendlyName),
                "serial": _decode_wmi_chars(row.SerialNumberID),
            }
        return result
    except Exception:
        return {}
    finally:
        row = None
        rows = None
        service = None
        pythoncom.CoUninitialize()


def _enrich_linux(displays: list[DisplayInfo]) -> list[DisplayInfo]:
    try:
        output = subprocess.check_output(
            ["xrandr", "--listactivemonitors"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return displays
    pattern = re.compile(r"^\s*\d+:\s+[^ ]*\s+(\d+)/\d+x(\d+)/\d+([+-]\d+)([+-]\d+)\s+(.+)$")
    metadata = []
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        width, height, left, top, name = match.groups()
        metadata.append((int(left), int(top), int(width), int(height), name.strip()))
    return [
        replace(
            display,
            stable_id=f"xrandr:{item[4]}",
            name=item[4],
            model=item[4],
        )
        if (item := _metadata_by_geometry(metadata, display)) is not None
        else display
        for display in displays
    ]


def _enrich_macos(displays: list[DisplayInfo]) -> list[DisplayInfo]:
    try:
        import AppKit
        import Quartz
    except Exception:
        return displays
    metadata = []
    for screen in AppKit.NSScreen.screens():
        display_id = int(screen.deviceDescription().get("NSScreenNumber", 0))
        bounds = Quartz.CGDisplayBounds(display_id)
        left, top = int(bounds.origin.x), int(bounds.origin.y)
        width, height = int(bounds.size.width), int(bounds.size.height)
        name = str(getattr(screen, "localizedName", lambda: "")() or "").strip() or None
        vendor = int(Quartz.CGDisplayVendorNumber(display_id))
        model_number = int(Quartz.CGDisplayModelNumber(display_id))
        serial = int(Quartz.CGDisplaySerialNumber(display_id))
        stable_id = f"cg:{vendor}:{model_number}:{serial or display_id}"
        metadata.append((left, top, width, height, stable_id, name, vendor, model_number, serial))
    enriched = []
    for display in displays:
        item = _metadata_by_geometry(metadata, display)
        if item is None:
            enriched.append(display)
            continue
        enriched.append(
            replace(
                display,
                stable_id=item[4],
                name=item[5],
                manufacturer=str(item[6]),
                model=item[5] or str(item[7]),
                serial=str(item[8]) if item[8] else None,
            )
        )
    return enriched


def _assign_display_numbers(displays: list[DisplayInfo], platform_name: str) -> list[DisplayInfo]:
    if platform_name == "Windows":
        ordered = sorted(
            displays,
            key=lambda item: item.device_display_number or item.capture_index,
        )
    else:
        ordered = sorted(displays, key=lambda item: item.display_number)
    return [replace(display, display_number=index) for index, display in enumerate(ordered, start=1)]


def _display_by_geometry(
    displays: Iterable[DisplayInfo], left: int, top: int, width: int, height: int
) -> DisplayInfo | None:
    return next(
        (
            display
            for display in displays
            if display.left == left
            and display.top == top
            and display.width == width
            and display.height == height
        ),
        None,
    )


def _metadata_by_geometry(metadata: Iterable[tuple], display: DisplayInfo) -> tuple | None:
    return next(
        (
            item
            for item in metadata
            if tuple(item[:4]) == (display.left, display.top, display.width, display.height)
        ),
        None,
    )


def _windows_instance_key(value: str | None) -> str:
    text = str(value or "").upper().replace("#", "\\")
    marker = "DISPLAY\\"
    if marker in text:
        text = text[text.index(marker):]
    text = text.split("\\{", 1)[0].removesuffix("_0")
    return text


def _windows_display_number(device_name: str) -> int | None:
    suffix = str(device_name).upper().rsplit("DISPLAY", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _decode_wmi_chars(values: Any) -> str | None:
    text = "".join(chr(int(value)) for value in values if int(value)).strip()
    return text or None


def _normalized_text(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def _clamp_monitor_index(monitor_index: int, monitor_count: int) -> int:
    return min(max(0, int(monitor_index) - 1), max(0, int(monitor_count) - 1))


def _platform_name() -> str:
    if os.name == "nt":
        return "Windows"
    if sys_platform := os.environ.get("D2S_PLATFORM_NAME"):
        return sys_platform
    import platform

    return platform.system()
