import numpy as np

try:
    import xr
except ImportError:
    xr = None

from .xr_math import _xr_quat_to_mat4


class CoreControllerPoseMixin:
    def _update_aim_poses(self, display_time):
        """Locate both controller aim spaces and cache their world-space 4x4 matrices."""
        now = self._frame_now
        for space, mat_attr, prev_attr, move_attr in [
            (self._aim_space_l, "_aim_mat_l", "_laser_prev_mat_l", "_laser_last_move_l"),
            (self._aim_space_r, "_aim_mat_r", "_laser_prev_mat_r", "_laser_last_move_r"),
        ]:
            if space is None:
                setattr(self, mat_attr, None)
                continue
            try:
                loc = xr.locate_space(space, self._xr_space, display_time)
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    R = _xr_quat_to_mat4(loc.pose.orientation)
                    R[:3, 3] = [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z]
                    setattr(self, mat_attr, R)
                    # Compare position+orientation to previous pose to detect motion
                    prev = getattr(self, prev_attr)
                    if prev is not None:
                        pos_delta = float(np.linalg.norm(R[:3, 3] - prev[:3, 3]))
                        # Rotation difference via Frobenius norm of delta rotation matrix
                        rot_delta = float(np.linalg.norm(R[:3, :3] - prev[:3, :3]))
                        if pos_delta > self._LASER_MOVE_THRESH or rot_delta > self._LASER_MOVE_THRESH:
                            setattr(self, move_attr, now)
                    else:
                        setattr(self, move_attr, now)
                    setattr(self, prev_attr, R.copy())
                else:
                    setattr(self, mat_attr, None)
            except Exception:
                setattr(self, mat_attr, None)

    def _update_grip_poses(self, display_time):
        """Locate controller grip spaces and cache 4x4 world-space matrices.
        Controller 3D models are placed at the grip center (aim pose is at the tracking ring).
        Also update movement timestamps for 5-second idle auto-hide."""
        now = self._frame_now
        for space, mat_attr, move_attr in [
            (self._grip_space_l, "_grip_mat_l", "_laser_last_move_l"),
            (self._grip_space_r, "_grip_mat_r", "_laser_last_move_r"),
        ]:
            if space is None:
                setattr(self, mat_attr, None)
                continue
            try:
                loc = xr.locate_space(space, self._xr_space, display_time)
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    R = _xr_quat_to_mat4(loc.pose.orientation)
                    R[:3, 3] = [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z]
                    # Detect movement
                    prev = getattr(self, mat_attr)
                    if prev is not None:
                        delta = float(np.linalg.norm(R[:3, 3] - prev[:3, 3]))
                        if delta > self._LASER_MOVE_THRESH:
                            setattr(self, move_attr, now)
                    else:
                        setattr(self, move_attr, now)
                    setattr(self, mat_attr, R)
                else:
                    setattr(self, mat_attr, None)
            except Exception:
                setattr(self, mat_attr, None)

    def _reset_orientation_offsets(self):
        """Clear manual yaw/pitch offsets so screen faces head baseline."""
        self._yaw_offset   = 0.0
        self._pitch_offset = 0.0

    def _clear_screen_grab_anchors(self):
        """Drop stale grip anchors after a screen teleport/reset."""
        self._screen_grab_local_l = None
        self._screen_grab_local_r = None
        self._screen_grab_grip_l = None
        self._screen_grab_grip_r = None
        self._kb_grab_local_l = None
        self._kb_grab_local_r = None
