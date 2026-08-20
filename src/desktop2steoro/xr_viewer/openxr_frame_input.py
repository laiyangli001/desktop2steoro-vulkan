try:
    import xr
except ImportError:
    xr = None


class OpenXRFrameInput:
    def __init__(self, viewer):
        self.viewer = viewer

    def sync_actions(self):
        viewer = self.viewer
        if viewer._xr_actions_sync_info is None:
            return
        try:
            xr.sync_actions(viewer._xr_session, viewer._xr_actions_sync_info)
        except Exception:
            pass

    def update_controller_frame(self, *, display_time, dt):
        viewer = self.viewer
        viewer._update_aim_poses(display_time)
        viewer._update_grip_poses(display_time)

        both_missing = viewer._aim_mat_l is None and viewer._aim_mat_r is None
        if both_missing:
            viewer._controller_miss_frames += 1
        else:
            viewer._controller_miss_frames = 0

        if viewer._controller_miss_frames < 30:
            viewer._smooth_controller_poses()
            viewer._update_trackpad_button_emu()
            viewer._poll_controller_input(dt)
            return 'controller_input'

        viewer._emu_y = viewer._emu_x = viewer._emu_b = False
        viewer._emu_a = viewer._emu_lsc = viewer._emu_rsc = False
        viewer._cursor_uv_l = None
        viewer._cursor_uv_r = None
        viewer._cursor_ctrl = None
        viewer._cursor_smooth_uv = None
        viewer._grabbed = False
        return 'controller_missing'
