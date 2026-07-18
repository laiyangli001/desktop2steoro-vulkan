from __future__ import annotations


def monitor_contains(mon, rect):
    left, top, w, h = rect
    right, bottom = left + w, top + h
    mon_left, mon_top = mon["left"], mon["top"]
    mon_right, mon_bottom = mon_left + mon["width"], mon_top + mon["height"]
    return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom


def monitor_intersection_area(mon, rect):
    left, top, w, h = rect
    right, bottom = left + w, top + h
    mon_left, mon_top = mon["left"], mon["top"]
    mon_right, mon_bottom = mon_left + mon["width"], mon_top + mon["height"]
    inter_w = max(0, min(mon_right, right) - max(mon_left, left))
    inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
    return inter_w * inter_h


def choose_monitor_and_rect(monitors, rect):
    for mon in monitors:
        if monitor_contains(mon, rect):
            return mon, rect

    best_mon, best_area = None, -1
    for mon in monitors:
        area = monitor_intersection_area(mon, rect)
        if area > best_area:
            best_area = area
            best_mon = mon

    if best_mon is None or best_area <= 0:
        best_mon = monitors[0]

    left, top, w, h = rect
    right, bottom = left + w, top + h
    mon_left, mon_top = best_mon["left"], best_mon["top"]
    mon_right, mon_bottom = mon_left + best_mon["width"], mon_top + best_mon["height"]
    new_left = max(left, mon_left)
    new_top = max(top, mon_top)
    new_right = min(right, mon_right)
    new_bottom = min(bottom, mon_bottom)
    new_w = max(0, new_right - new_left)
    new_h = max(0, new_bottom - new_top)

    if new_w == 0 or new_h == 0:
        return best_mon, (mon_left, mon_top, best_mon["width"], best_mon["height"])

    return best_mon, (new_left, new_top, new_w, new_h)
