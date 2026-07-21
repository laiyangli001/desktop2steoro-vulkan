import ctypes
import math
import time

import numpy as np

from .windows_input import (
    _KEYEVENTF_KEYUP,
    _send_hscroll,
    _send_key,
    _send_vscroll,
)


class CoreInputHelpersMixin:
    def _send_arrow_impl(self, value, neg_dir, pos_dir):
        """Send arrow key hold/release based on stick value. Only fires on edge transitions."""
        VK_MAP = {'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27}
        neg_attr = f'_arrow_{neg_dir}_held'
        pos_attr = f'_arrow_{pos_dir}_held'
        neg_vk = VK_MAP[neg_dir]
        pos_vk = VK_MAP[pos_dir]

        if abs(value) <= self._input_deadzone():
            for attr in (neg_attr, pos_attr):
                if getattr(self, attr):
                    vk = neg_vk if attr == neg_attr else pos_vk
                    ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
                    setattr(self, attr, False)
            return

        if value < -self._input_deadzone():
            if not getattr(self, neg_attr):
                ctypes.windll.user32.keybd_event(neg_vk, 0, 0, 0)
                setattr(self, neg_attr, True)
            if getattr(self, pos_attr):
                ctypes.windll.user32.keybd_event(pos_vk, 0, _KEYEVENTF_KEYUP, 0)
                setattr(self, pos_attr, False)
        elif value > self._input_deadzone():
            if not getattr(self, pos_attr):
                ctypes.windll.user32.keybd_event(pos_vk, 0, 0, 0)
                setattr(self, pos_attr, True)
            if getattr(self, neg_attr):
                ctypes.windll.user32.keybd_event(neg_vk, 0, _KEYEVENTF_KEYUP, 0)
                setattr(self, neg_attr, False)

    def _press_key_impl(self, key, key_idx, held_key_attr, held_mods_attr):
        """Press and hold a regular key on the virtual keyboard (key-down only)."""
        kbd = ctypes.windll.user32.keybd_event
        VK_SHIFT = 0x10
        VK_CTRL = 0x11
        VK_ALT = 0x12
        VK_WIN = 0x5B
        sh = self._mod_state['shift']
        ct = self._mod_state['ctrl']
        al = self._mod_state['alt']
        wn = self._mod_state['win']
        shift_on = sh[0] or sh[1]
        ctrl_on = ct[0] or ct[1]
        alt_on = al[0] or al[1]
        win_on = wn[0] or wn[1]
        use_shift = shift_on ^ self._caps_lock
        vk_to_use = key.shifted_vk if use_shift else key.vk
        need_shift = use_shift and vk_to_use == key.vk
        if ctrl_on:
            kbd(VK_CTRL, 0, 0, 0)
        if need_shift:
            kbd(VK_SHIFT, 0, 0, 0)
        if alt_on:
            kbd(VK_ALT, 0, 0, 0)
        if win_on:
            kbd(VK_WIN, 0, 0, 0)
        kbd(vk_to_use, 0, 0, 0)
        setattr(self, held_key_attr, key_idx)
        setattr(self, held_mods_attr, (need_shift, ctrl_on, alt_on, win_on, vk_to_use))

    def _handle_keyboard_input(self):
        """Send Windows keystrokes when a controller trigger fires on a keyboard key."""
        if not self._keyboard_visible:
            self._kb_hover_l = None
            self._kb_hover_r = None
            return
        CLICK_THRESH = 0.7
        RELEASE_THRESH = 0.3
        VK_SHIFT = 0x10
        VK_CAPS = 0x14
        VK_CTRL = 0x11
        VK_ALT = 0x12
        VK_WIN = 0x5B
        kbd = ctypes.windll.user32.keybd_event

        gripping = bool(self._grip_l_now or self._grip_r_now)

        if gripping:
            lt = 0.0
            rt = 0.0
        else:
            lt = self._read_float_action(self._act_left_trigger, "/user/hand/left")
            rt = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        HOVER_DEBOUNCE = 0
        for trig_now, trig_prev_attr, hover_attr, held_key_attr, held_mods_attr, aim_mat, debounce_attr in [
            (lt, '_kb_trig_prev_l', '_kb_hover_l', '_kb_held_key_l', '_kb_held_mods_l', self._aim_mat_l, '_kb_debounce_l'),
            (rt, '_kb_trig_prev_r', '_kb_hover_r', '_kb_held_key_r', '_kb_held_mods_r', self._aim_mat_r, '_kb_debounce_r'),
        ]:
            trig_prev = getattr(self, trig_prev_attr)
            held_key = getattr(self, held_key_attr)
            held_mods = getattr(self, held_mods_attr)
            if aim_mat is not None:
                grip_mat = self._grip_mat_l if aim_mat is self._aim_mat_l else self._grip_mat_r
                fw = -aim_mat[:3, 2].astype('f8')
                right = aim_mat[:3, 0].astype('f8')
                ang = math.radians(12)
                ca = math.cos(ang)
                sa = math.sin(ang)
                axis = right / (np.linalg.norm(right) + 1e-10)
                fw = fw * ca + np.cross(axis, fw) * sa + axis * np.dot(axis, fw) * (1 - ca)
                if grip_mat is not None:
                    cp = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    cp = aim_mat[:3, 3].astype('f8')
                cp = cp + fw * 0.11
                lx_ly = self._keyboard_plane_hit(cp, fw)
                raw_idx = None
                if lx_ly[0] is not None:
                    smooth_attr = '_kb_smooth_l' if hover_attr.endswith('_l') else '_kb_smooth_r'
                    prev_smooth = getattr(self, smooth_attr, None)
                    if prev_smooth is None:
                        smooth = np.array([lx_ly[0], lx_ly[1]], dtype='f8')
                    else:
                        alpha = 0.3
                        smooth = prev_smooth + (np.array([lx_ly[0], lx_ly[1]], dtype='f8') - prev_smooth) * alpha
                    setattr(self, smooth_attr, smooth)
                    sx, sy = float(smooth[0]), float(smooth[1])
                    for key_index, key in enumerate(self._keyboard_keys):
                        x0, y0, x1, y1 = key.rect_local
                        if x0 <= sx <= x1 and y0 <= sy <= y1:
                            raw_idx = key_index
                            break
                else:
                    setattr(self, '_kb_smooth_l' if hover_attr.endswith('_l') else '_kb_smooth_r', None)
            else:
                raw_idx = None

            prev_hover = getattr(self, hover_attr)
            if raw_idx == prev_hover:
                setattr(self, debounce_attr, 0)
                idx = raw_idx
            else:
                count = getattr(self, debounce_attr, 0) + 1
                setattr(self, debounce_attr, count)
                if count >= HOVER_DEBOUNCE:
                    idx = raw_idx
                else:
                    idx = prev_hover
            setattr(self, hover_attr, idx)
            if idx is not None and idx != prev_hover and not gripping:
                hand_path = "/user/hand/left" if hover_attr.endswith('_l') else "/user/hand/right"
                self._pulse_haptic(hand_path, amplitude=0.18, duration_s=0.018, min_interval_s=0.045)

            if held_key is not None:
                release = False
                if trig_now < RELEASE_THRESH:
                    release = True
                elif idx != held_key:
                    release = True
                if release:
                    shift_dn, ctrl_dn, alt_dn, win_dn, vk_held = held_mods
                    kbd(vk_held, 0, _KEYEVENTF_KEYUP, 0)
                    if win_dn:
                        kbd(VK_WIN, 0, _KEYEVENTF_KEYUP, 0)
                    if alt_dn:
                        kbd(VK_ALT, 0, _KEYEVENTF_KEYUP, 0)
                    if shift_dn:
                        kbd(VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0)
                    if ctrl_dn:
                        kbd(VK_CTRL, 0, _KEYEVENTF_KEYUP, 0)
                    for name in ('shift', 'ctrl', 'alt', 'win'):
                        self._mod_state[name][0] = False
                    setattr(self, held_key_attr, None)
                    setattr(self, held_mods_attr, None)
                    held_key = None

            if trig_now >= CLICK_THRESH and trig_prev < CLICK_THRESH and idx is not None:
                key = self._keyboard_keys[idx]
                mod_name = {VK_SHIFT: 'shift', VK_CTRL: 'ctrl', VK_ALT: 'alt', VK_WIN: 'win'}.get(key.vk)
                if key.vk in (0x25, 0x26, 0x27, 0x28):
                    adjust_frosted = getattr(self, '_adjust_frosted_glow_vk', None)
                    if callable(adjust_frosted) and adjust_frosted(key.vk):
                        setattr(self, trig_prev_attr, trig_now)
                        continue
                if mod_name is not None:
                    double_tap_window = 0.4
                    now_t = time.monotonic()
                    state = self._mod_state[mod_name]
                    if state[1]:
                        state[0] = False
                        state[1] = False
                    elif state[0]:
                        state[0] = False
                        _send_key(key.vk)
                    elif (now_t - state[2]) < double_tap_window:
                        state[0] = False
                        state[1] = True
                    else:
                        state[0] = True
                    state[2] = now_t
                elif key.vk == VK_CAPS:
                    self._caps_lock = not self._caps_lock
                else:
                    self._press_key(key, idx, held_key_attr, held_mods_attr)

            if held_key is None and trig_now >= CLICK_THRESH and idx is not None and trig_prev >= CLICK_THRESH:
                key = self._keyboard_keys[idx]
                if key.vk not in (VK_SHIFT, VK_CTRL, VK_ALT, VK_WIN, VK_CAPS):
                    self._press_key(key, idx, held_key_attr, held_mods_attr)

            setattr(self, trig_prev_attr, trig_now)

        sh = self._mod_state['shift']
        cur_shifted = bool(sh[0] or sh[1] or self._caps_lock)
        if cur_shifted != self._kb_show_shifted:
            self._kb_show_shifted = cur_shifted
            self._refresh_or_upload_keyboard_content()

    def _accum_scroll(self, x_axis, y_axis, dt):
        """Accumulate thumbstick deflection into accelerated mouse wheel events."""
        WHEEL_DELTA = 100
        SCROLL_BASE_NOTCH = 2.0
        SCROLL_MAX_NOTCH = 35.0
        ACCEL_EXPONENT = 2.8

        for axis_val, accum_attr, send_fn in [
            (x_axis, '_scroll_accum_x', _send_hscroll),
            (y_axis, '_scroll_accum_y', _send_vscroll),
        ]:
            mag = abs(axis_val)
            if mag <= self._input_deadzone():
                continue
            t = (mag - self._input_deadzone()) / (1.0 - self._input_deadzone())
            speed = SCROLL_BASE_NOTCH + (SCROLL_MAX_NOTCH - SCROLL_BASE_NOTCH) * (t ** ACCEL_EXPONENT)
            accum = getattr(self, accum_attr) + float(axis_val) * speed * dt
            whole = int(accum)
            if whole:
                send_fn(whole * WHEEL_DELTA)
                accum -= whole
            setattr(self, accum_attr, accum)

    def _input_deadzone(self):
        return DEAD


DEAD = 0.15
