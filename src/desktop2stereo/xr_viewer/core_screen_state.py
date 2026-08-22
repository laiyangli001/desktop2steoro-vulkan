import math

import numpy as np


class CoreScreenStateMixin:
    def _screen_view_distance(self):
        if getattr(self, '_head_pos_w', None) is None:
            return float(self.screen_distance)
        screen_pos = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance], dtype='f8')
        head_pos = np.asarray(self._head_pos_w, dtype='f8')
        return float(np.linalg.norm(screen_pos - head_pos))

    def _screen_pose_mat4(self):
        """Screen rigid transform without width/height scale."""
        cy = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch)
        sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll)
        sr = math.sin(self.screen_roll)
        rot_y = np.array([[cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
        rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
        rot_z = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
        trans = np.eye(4, dtype='f4')
        trans[0, 3] = self.screen_pan_x
        trans[1, 3] = self.screen_pan_y
        trans[2, 3] = -self.screen_distance
        return trans @ rot_y @ rot_x @ rot_z

    def _screen_model_mat4(self):
        """Screen transform used by projection rendering."""
        self._ensure_screen_dimensions()
        scale = np.diag([
            self.screen_width / 2.0,
            self.screen_height / 2.0,
            1.0,
            1.0,
        ]).astype('f4')
        return self._screen_pose_mat4() @ scale

    def _decompose_env_model_mat4(self, mat):
        """Store an environment model matrix back into position, rotation, and scale."""
        mat = np.asarray(mat, dtype=np.float64)
        self._env_model_pos = [float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])]
        rot_scale = mat[:3, :3].copy()
        scale = [float(np.linalg.norm(rot_scale[:, i])) for i in range(3)]
        for i, s in enumerate(scale):
            if s > 1e-9:
                rot_scale[:, i] /= s
        pitch = math.asin(max(-1.0, min(1.0, -float(rot_scale[1, 2]))))
        yaw = math.atan2(float(rot_scale[0, 2]), float(rot_scale[2, 2]))
        roll = math.atan2(float(rot_scale[1, 0]), float(rot_scale[1, 1]))
        self._env_model_rot = [yaw, pitch, roll]
        self._env_model_scale = scale
        if hasattr(self, '_cached_env_model_mat4_frame'):
            self._cached_env_model_mat4_frame = -1

    def _move_env_with_screen_delta(self, old_screen_mat):
        """When a locked room screen is reset, apply the same rigid delta to the room."""
        if not self._environment_screen_locked() or old_screen_mat is None:
            return
        try:
            new_screen_mat = self._screen_pose_mat4().astype('f8')
            delta = new_screen_mat @ np.linalg.inv(old_screen_mat.astype('f8'))
            env_new = delta @ self._build_env_model_mat4().astype('f8')
            self._decompose_env_model_mat4(env_new)
        except Exception as exc:
            print(f"[OpenXRViewer] screen/env lock transform failed: {exc}")

    def _reset_locked_environment_to_profile(self, show_border=False):
        """Restore a locked room's calibrated environment/screen pose."""
        if not getattr(self, '_active_environment', None):
            return False
        return self._apply_profile_screen_layout(show_border=show_border)

    def _kb_restore_cached_position(self, cached):
        """Restore keyboard position cached before an environment switch."""
        self._keyboard_pan_x = float(cached['pan_x'])
        self._keyboard_pan_y = float(cached['pan_y'])
        self._keyboard_distance = float(cached['distance'])
        self._keyboard_width = float(cached.get('width', self.screen_width * 0.75))
        self._keyboard_yaw = float(cached.get('yaw', 0.0))
        self._keyboard_pitch = float(cached.get('pitch', 0.0))

    def _tick_screen_anim(self, dt):
        """Advance an optional screen reset animation target."""
        if getattr(self, '_anim_target_pan_x', None) is None and getattr(self, '_anim_target_roll', None) is None:
            return
        k = 6.0
        alpha = 1.0 - math.exp(-k * max(dt, 1e-4))

        def _lerp(a, b):
            return a + alpha * (b - a)

        def _lerp_angle(a, b):
            d = (b - a + math.pi) % (2 * math.pi) - math.pi
            return a + alpha * d

        self.screen_pan_x = _lerp(self.screen_pan_x, self._anim_target_pan_x)
        self.screen_pan_y = _lerp(self.screen_pan_y, self._anim_target_pan_y)
        self.screen_distance = _lerp(self.screen_distance, self._anim_target_distance)
        self.screen_yaw = _lerp_angle(self.screen_yaw, self._anim_target_yaw)
        self.screen_pitch = _lerp_angle(self.screen_pitch, self._anim_target_pitch)
        self.screen_roll = _lerp_angle(self.screen_roll, self._anim_target_roll)
        close = (
            abs(self.screen_pan_x - self._anim_target_pan_x) < 0.001
            and abs(self.screen_pan_y - self._anim_target_pan_y) < 0.001
            and abs(self.screen_distance - self._anim_target_distance) < 0.001
            and abs((self.screen_yaw - self._anim_target_yaw + math.pi) % (2 * math.pi) - math.pi) < 0.0002
            and abs((self.screen_pitch - self._anim_target_pitch + math.pi) % (2 * math.pi) - math.pi) < 0.0002
            and abs((self.screen_roll - self._anim_target_roll + math.pi) % (2 * math.pi) - math.pi) < 0.0002
        )
        if close:
            self.screen_pan_x = self._anim_target_pan_x
            self.screen_pan_y = self._anim_target_pan_y
            self.screen_distance = self._anim_target_distance
            self.screen_yaw = self._anim_target_yaw
            self.screen_pitch = self._anim_target_pitch
            self.screen_roll = self._anim_target_roll
            self._anim_target_pan_x = None
            self._anim_target_pan_y = None
            self._anim_target_distance = None
            self._anim_target_yaw = None
            self._anim_target_pitch = None
            self._anim_target_roll = None
