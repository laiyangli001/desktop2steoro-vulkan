import json
import math
import os
import time

import numpy as np

try:
    import xr
except ImportError:
    xr = None

from .xr_math import euler_to_mat4, mat4_to_xr_posef


class CoreScreenControlMixin:
    def _reset_screen_to_gaze(self, show_border=False):
        """Snap the screen to two meters in front of current head gaze."""
        reset_dist = 2.0
        old_screen_mat = self._screen_pose_mat4() if self._environment_screen_locked() else None
        self._reset_orientation_offsets()
        self._clear_screen_grab_anchors()
        self._anim_target_pan_x = None
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            flen = math.sqrt(fx * fx + fy * fy + fz * fz)
            if flen > 1e-4:
                fx /= flen
                fy /= flen
                fz /= flen
            else:
                fx, fy, fz = 0.0, 0.0, -1.0
            tx = hx + fx * reset_dist
            ty = hy + fy * reset_dist
            tz = hz + fz * reset_dist
            horiz = math.sqrt(fx * fx + fz * fz)
            yaw = math.atan2(-fx, -fz) if horiz > 1e-4 else self.screen_yaw
            pitch = math.asin(max(-0.999, min(0.999, fy)))
            cy = math.cos(yaw)
            sy_ = math.sin(yaw)
            cp = math.cos(pitch)
            sp = math.sin(pitch)
            x1 = cy * tx - sy_ * tz
            y1 = ty
            z1 = sy_ * tx + cy * tz
            self.screen_pan_x = x1
            self.screen_pan_y = cp * y1 + sp * z1
            self.screen_distance = -(-sp * y1 + cp * z1)
            self.screen_yaw = yaw
            self.screen_pitch = pitch
            self.screen_roll = 0.0
        else:
            self.screen_distance = reset_dist
            self.screen_pan_x = 0.0
            self.screen_pan_y = float(self._initial_head_y)
            self.screen_pitch = 0.0
            self.screen_yaw = 0.0
            self.screen_roll = 0.0
        self._move_env_with_screen_delta(old_screen_mat)
        if show_border:
            self._border_alpha = 1.0
            self._border_idle_t = time.perf_counter()
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()

    def _reset_screen_direction(self):
        """Turn the screen to face current head gaze while preserving distance."""
        if self._head_pos_w is None or self._head_fwd_w is None:
            return
        old_screen_mat = self._screen_pose_mat4() if self._environment_screen_locked() else None
        hx, hy, hz = self._head_pos_w
        fx, fy, fz = self._head_fwd_w
        flen = math.sqrt(fx * fx + fy * fy + fz * fz)
        if flen > 1e-4:
            fx /= flen
            fy /= flen
            fz /= flen
        else:
            fx, fy, fz = 0.0, 0.0, -1.0
        dx = self.screen_pan_x - hx
        dy = self.screen_pan_y - hy
        dz = -self.screen_distance - hz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        self._reset_orientation_offsets()
        self._clear_screen_grab_anchors()
        self._anim_target_pan_x = None
        self.screen_pan_x = hx + fx * dist
        self.screen_pan_y = hy + fy * dist
        self.screen_distance = -(hz + fz * dist)
        self.screen_yaw = math.atan2(-fx, -fz)
        self.screen_pitch = math.asin(max(-0.999, min(0.999, fy)))
        self.screen_roll = 0.0
        self._move_env_with_screen_delta(old_screen_mat)
        self._border_alpha = 1.0
        self._border_idle_t = time.perf_counter()
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()

    def _apply_preset(self, index):
        """Apply screen preset: size, distance, and reposition to face the user."""
        if not self._screen_presets:
            return False
        index = int(index) % len(self._screen_presets)
        name, width_m, distance_m = self._screen_presets[index]
        previous_screen_width = float(self.screen_width)
        self._reset_orientation_offsets()
        self._clear_screen_grab_anchors()
        self.screen_width = float(width_m)
        self._screen_ref_size = float(width_m)
        self.screen_height = None
        self.screen_pitch = 0.0
        self.screen_roll = 0.0
        self._anim_target_pan_x = None
        self._anim_target_pan_y = None
        self._anim_target_distance = None
        self._anim_target_yaw = None
        self._anim_target_pitch = None
        self._anim_target_roll = None
        self._screen_curved = False
        self._preset_index = index
        self.screen_pan_y = float(self._initial_head_y)

        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, _, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            flen = math.sqrt(fx * fx + fy * fy + fz * fz)
            if flen > 1e-4:
                fx /= flen
                fy /= flen
                fz /= flen
            else:
                fx, fy, fz = 0.0, 0.0, -1.0
            self.screen_pan_x = hx + fx * float(distance_m)
            self.screen_distance = -(hz + fz * float(distance_m))
            self.screen_yaw = math.atan2(-fx, -fz)
        else:
            self.screen_pan_x = 0.0
            self.screen_distance = float(distance_m)
            self.screen_yaw = 0.0

        display_height_m = float(width_m) * 9.0 / 16.0
        view_distance = self._screen_view_distance()
        self._preset_name_overlay = (
            f"{name}  {float(width_m):.2f} x {display_height_m:.2f} m"
            f"  @ {view_distance:.2f} m"
        )
        self._preset_osd_last_key = None
        self._last_overlay_update = 0.0
        self._screen_footprint_logged.clear()
        self._border_alpha = 1.0
        self._border_idle_t = time.perf_counter()
        if self._keyboard_visible:
            self._sync_keyboard_size_from_screen_width(previous_screen_width)
            self._anchor_keyboard_below_screen()
        print(
            f"[OpenXRViewer] Screen preset: {name} "
            f"width={self.screen_width:.3f}m distance={view_distance:.3f}m",
            flush=True,
        )
        return True

    def _cycle_screen_preset(self):
        return self._apply_preset(self._preset_index + 1)

    def _reset_seating_vertical(self):
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return
        if 'y' in view:
            x, y, z, angle = self._seat_adjust_current_pos()
            self._apply_seat_adjust_xr_space(x, y, z, angle)
            return
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return
        raw_head = self._head_model_mat4_from_views(views)
        if raw_head is None:
            return
        auto_center = bool(view.get('auto_center_on_screen', False))
        pos_keys = ('position', 'camera_position', 'viewer_position')
        rot_deg_keys = ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg')
        rot_rad_keys = ('rotation', 'camera_rotation', 'viewer_rotation')
        has_rot = any(key in view for key in rot_deg_keys + rot_rad_keys)
        has_pos = any(key in view for key in pos_keys)
        if auto_center:
            auto_pos = self._auto_view_position_from_screen(view, has_rot, rot_deg_keys, rot_rad_keys)
            if auto_pos is None and has_pos:
                auto_pos = self._profile_vec3(view, pos_keys, [0.0, 0.0, 0.0])
            if auto_pos is None:
                return
            desired_y = float(auto_pos[1])
        elif has_pos:
            desired_y = float(self._profile_vec3(view, pos_keys, [0.0, 0.0, 0.0])[1])
        else:
            return
        current_space = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
        native_head_y = float((current_space @ raw_head)[1, 3])
        dy = native_head_y - desired_y
        new_space_in_ref = current_space.copy()
        new_space_in_ref[1, 3] = dy
        try:
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=mat4_to_xr_posef(new_space_in_ref),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Vertical reseat failed: {exc}")
            return
        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = new_space_in_ref
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass
        print(f"[OpenXRViewer] Vertical reseat: desired_y={desired_y:.3f} native_y={native_head_y:.3f}")

    def _seat_adjust_current_pos(self):
        view = getattr(self, '_view_pose_profile', {}) or {}
        try:
            x = float(view.get('x', 0.0))
            y = float(view.get('y', 0.6))
            z = float(view.get('z', 0.0))
            angle = float(view.get('angle', 0.0))
        except (TypeError, ValueError):
            x, y, z, angle = 0.0, 0.6, 0.0, 0.0
        return x, y, z, angle

    def _enter_seat_adjust_mode(self):
        if self._seat_adjust_active:
            return
        self._seat_adjust_active = True
        self._seat_adjust_t = time.perf_counter()
        self._seat_adjust_osd_show_t = time.perf_counter()
        self._seat_adjust_osd_dirty = True
        self._seat_adjust_grip_move = False
        print("[OpenXRViewer] Entered seat adjust mode")

    def _exit_seat_adjust_mode(self, save=True):
        if not self._seat_adjust_active:
            return
        self._seat_adjust_active = False
        if save:
            self._save_view_pose_to_profile()
        self._seat_adjust_osd_dirty = True
        print("[OpenXRViewer] Exited seat adjust mode (saved=%s)" % save)

    def _apply_seat_adjust_xr_space(self, x, y, z, angle):
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return
        position = screen.get('position', screen.get('screen_position'))
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return
        try:
            screen_pos = np.array([float(position[0]), float(position[1]), float(position[2])], dtype=np.float32)
        except (TypeError, ValueError):
            return
        screen_rot = self._profile_rotation_rad(
            screen,
            ('rotation_deg', 'screen_rotation_deg'),
            ('rotation', 'screen_rotation'),
            [0.0, 0.0, 0.0],
        )
        screen_mat = euler_to_mat4(*screen_rot).astype(np.float32)
        screen_normal = screen_mat[:3, 2].copy()
        screen_right = screen_mat[:3, 0].copy()
        screen_up = screen_mat[:3, 1].copy()
        viewer_pos = screen_pos + screen_right * float(x) + screen_normal * float(y) + screen_up * float(z)
        angle_rad = math.radians(float(angle))
        face_screen_yaw = math.atan2(float(screen_normal[0]), float(screen_normal[2]))
        viewer_yaw = face_screen_yaw + angle_rad
        desired_head = euler_to_mat4(viewer_yaw, 0.0, 0.0).astype(np.float32)
        desired_head[:3, 3] = viewer_pos
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return
        raw_head = self._head_model_mat4_from_views(views)
        if raw_head is None:
            return
        current_space_in_ref = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
        reference_head = current_space_in_ref @ raw_head
        leveled = self._level_head_model_mat4(reference_head)
        if leveled is not None:
            reference_head = leveled
        space_in_ref = reference_head @ np.linalg.inv(desired_head)
        try:
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=mat4_to_xr_posef(space_in_ref.astype(np.float32)),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Seat adjust XR space failed: {exc}")
            return
        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = space_in_ref.astype(np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass

    def _save_view_pose_to_profile(self):
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict):
            return
        env_name = getattr(self, '_active_environment', None) or getattr(self, '_environment_model', None)
        if not env_name or str(env_name).lower() in ('default', 'default glow', 'none'):
            return
        room_dir = os.path.join(self._environment_root, str(env_name))
        profile_path = os.path.join(room_dir, 'profile.json')
        if not os.path.exists(profile_path):
            return
        try:
            with open(profile_path, 'r', encoding='utf-8-sig') as f:
                profile = json.load(f)
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to read profile for save: {exc}")
            return
        vp = None
        view_poses = profile.get('view_poses')
        if isinstance(view_poses, list) and view_poses:
            idx = int(getattr(self, '_view_pose_index', 0)) % len(view_poses)
            if not isinstance(view_poses[idx], dict):
                view_poses[idx] = {}
            vp = view_poses[idx]
            profile['view_pose_index'] = idx
        if vp is None:
            if 'view_pose' not in profile or not isinstance(profile.get('view_pose'), dict):
                profile['view_pose'] = {}
            vp = profile['view_pose']
        vp['x'] = round(float(view.get('x', 0.0)), 4)
        vp['y'] = round(float(view.get('y', 0.6)), 4)
        vp['z'] = round(float(view.get('z', 0.0)), 4)
        vp['angle'] = round(float(view.get('angle', 0.0)), 1)
        profile['model_position'] = [round(float(v), 4) for v in self._env_model_pos]
        profile['screen_light_intensity'] = round(float(getattr(self, '_screen_light_intensity', 3.5)), 2)
        if 'screen' in profile and isinstance(profile['screen'], dict):
            profile['screen']['curved'] = bool(self._screen_curved)
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(
                f"[OpenXRViewer] Saved view_pose to {profile_path}: "
                f"x={vp['x']} y={vp['y']} z={vp['z']} angle={vp['angle']} curved={self._screen_curved}"
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to save profile: {exc}")

    def _reset_screen_to_default(self, show_border=False):
        """Reset screen to upright default: 2 m ahead horizontally, perpendicular to floor.

        Screen is always vertical (pitch=0) and faces the user's current horizontal
        forward direction. Centre height matches the headset eye height recorded at
        session start so the screen sits comfortably in front of the user.
        Called at session start and by the Y button.
        """
        if self._apply_profile_screen_layout(show_border=show_border):
            self._clear_screen_grab_anchors()
            return

        default_index = int(getattr(self, '_default_screen_preset_index', getattr(self, '_preset_index', 0)))
        default_width = float(getattr(self, 'screen_width', 16.0))
        default_dist = float(getattr(self, 'screen_distance', 16.0))
        presets = getattr(self, '_screen_presets', []) or []
        if presets:
            _name, width_m, distance_m = presets[default_index % len(presets)]
            default_width = float(width_m)
            default_dist = float(distance_m)
            self._preset_index = default_index % len(presets)
        reset_dist = default_dist
        old_screen_mat = self._screen_pose_mat4() if self._environment_screen_locked() else None
        self._reset_orientation_offsets()
        self._clear_screen_grab_anchors()
        self.screen_width = default_width
        self._screen_ref_size = default_width
        self.screen_height = None
        self.screen_pitch = 0.0
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, _, hz = self._head_pos_w
            fx, _, fz = self._head_fwd_w
            horiz = math.sqrt(fx * fx + fz * fz)
            if horiz > 1e-4:
                fx /= horiz
                fz /= horiz
            else:
                fx, fz = 0.0, -1.0
            self.screen_distance = -(hz + fz * reset_dist)
            self.screen_pan_x = hx + fx * reset_dist
            self.screen_pan_y = float(self._initial_head_y)
            self.screen_yaw = math.atan2(-fx, -fz)
        else:
            self.screen_distance = reset_dist
            self.screen_pan_x = 0.0
            self.screen_pan_y = float(self._initial_head_y)
            self.screen_yaw = 0.0
        if show_border:
            self._border_alpha = 1.0
            self._border_idle_t = time.perf_counter()
        self._move_env_with_screen_delta(old_screen_mat)
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()
