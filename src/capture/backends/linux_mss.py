from __future__ import annotations

import mss
import numpy as np

from Xlib import display, X
import cv2

def get_window_coords(title):
    d = display.Display()
    root = d.screen().root

    # Get all windows
    window_ids = root.get_full_property(
        d.intern_atom('_NET_CLIENT_LIST'),
        X.AnyPropertyType
    ).value

    # Pre-intern atoms for common properties
    net_wm_name = d.intern_atom('_NET_WM_NAME')
    utf8_string = d.intern_atom('UTF8_STRING')

    for window_id in window_ids:
        window = d.create_resource_object('window', window_id)

        # Try multiple ways to get the window name
        name = None
        try:
            # Try _NET_WM_NAME first (UTF-8)
            name_prop = window.get_full_property(net_wm_name, utf8_string)
            if name_prop:
                name = name_prop.value.decode('utf-8')
            else:
                # Fall back to WM_NAME
                name = window.get_wm_name()
                if isinstance(name, bytes):
                    name = name.decode('utf-8', errors='replace')
        except:
            continue
        if name and title in name:
            # Get absolute coordinates (accounting for window decorations)
            geom = window.get_geometry()
            pos = geom.root.translate_coords(window_id, 0, 0)
            return (pos.x, pos.y, geom.width, geom.height)
    return None

class DesktopGrabber:
    def __init__(self, output_resolution=1080, fps=60, window_title=None, capture_mode="Monitor", monitor_index=1):
        self.scaled_height = output_resolution
        self.fps = fps
        self.window_title = window_title
        self.capture_mode = capture_mode
        self._mss = mss.mss(with_cursor=True)
        self.prev_rect = None
        self.monitor_index = monitor_index

        if self.capture_mode == "Monitor":
            # Initialize with the selected monitor
            if self.monitor_index >= len(self._mss.monitors):
                self.monitor_index = 1
            mon = self._mss.monitors[self.monitor_index]
            self.left, self.top, self.width, self.height = mon['left'], mon['top'], mon['width'], mon['height']
        else:
            # Initialize with window coordinates
            bounds = get_window_coords(self.window_title)
            if bounds is None:
                raise RuntimeError(f"Window '{self.window_title}' not found")
            self.left, self.top, self.width, self.height = bounds

        self.scaled_width = round(self.width * self.scaled_height / self.height)

    def _monitor_contains(self, mon, rect):
        """
        Check whether a rectangle is completely inside a monitor's bounds.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon['left'], mon['top']
        mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
        return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

    def _monitor_intersection_area(self, mon, rect):
        """
        Compute the area of overlap between a rectangle and a monitor.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon['left'], mon['top']
        mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
        inter_w = max(0, min(mon_right, right) - max(mon_left, left))
        inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
        return inter_w * inter_h

    def _choose_monitor_and_rect(self, rect):
        """
        Select the best monitor for the window and clamp the rectangle to fit.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h

        # Check if the window is fully inside any secondary monitor (index >= 1)
        for mon in self._mss.monitors[1:]:
            if self._monitor_contains(mon, rect):
                return mon, rect

        # Find monitor with largest overlapping area
        best_mon, best_area = None, -1
        for mon in self._mss.monitors[1:]:
            area = self._monitor_intersection_area(mon, rect)
            if area > best_area:
                best_area = area
                best_mon = mon

        # Fallback to first non-primary monitor if no overlap
        if best_mon is None or best_area <= 0:
            best_mon = self._mss.monitors[1]

        # Clamp rectangle to monitor bounds
        mon_left, mon_top = best_mon['left'], best_mon['top']
        mon_right, mon_bottom = mon_left + best_mon['width'], mon_top + best_mon['height']
        new_left = max(left, mon_left)
        new_top = max(top, mon_top)
        new_right = min(right, mon_right)
        new_bottom = min(bottom, mon_bottom)
        new_w = max(0, new_right - new_left)
        new_h = max(0, new_bottom - new_top)

        # Default to full monitor if clamping results in empty area
        if new_w == 0 or new_h == 0:
            return best_mon, (mon_left, mon_top, best_mon['width'], best_mon['height'])

        return best_mon, (new_left, new_top, new_w, new_h)

    def _ensure_rect(self):
        if self.capture_mode != "Monitor":
            bounds = get_window_coords(self.window_title)
            if bounds is None:
                return False
            if bounds == self.prev_rect:
                return True
            self.prev_rect = bounds

            # Apply monitor clamping logic
            _, clamped_rect = self._choose_monitor_and_rect(bounds)
            self.left, self.top, self.width, self.height = clamped_rect
            self.scaled_width = round(self.width * self.scaled_height / self.height)
        return True

    def grab(self):
        if not self._ensure_rect():
            return None, self.scaled_height

        monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
        shot = self._mss.grab(monitor)
        arr = np.asarray(shot)
        return arr, self.scaled_height

    def stop(self):
        """Stop the capture and clean up resources."""
        if hasattr(self, '_mss'):
            try:
                self._mss.close()
            except:
                pass
