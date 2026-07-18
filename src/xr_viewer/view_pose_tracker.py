try:
    import xr
except ImportError:
    xr = None


class ViewPoseTracker:
    def __init__(self, viewer):
        self.viewer = viewer

    def locate_views(self, *, display_time):
        viewer = self.viewer
        views = self._locate_views(display_time)
        adjusted = viewer._apply_profile_view_pose_to_xr_space(views)
        if adjusted:
            views = self._locate_views(display_time)
            viewer._update_aim_poses(display_time)
            viewer._update_grip_poses(display_time)
        self._cache_head_pose(views)
        self._initialize_screen_once(views)
        return views, adjusted

    def _locate_views(self, display_time):
        viewer = self.viewer
        try:
            _view_state, views = xr.locate_views(
                viewer._xr_session,
                xr.ViewLocateInfo(
                    view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                    display_time=display_time,
                    space=viewer._xr_space,
                ),
            )
            return views
        except Exception:
            return [None, None]

    def _cache_head_pose(self, views):
        viewer = self.viewer
        if not views or views[0] is None or views[1] is None:
            return
        try:
            viewer._last_located_views = views
            head_mat = viewer._head_model_mat4_from_views(views)
            viewer._head_pos_w = (
                float(head_mat[0, 3]),
                float(head_mat[1, 3]),
                float(head_mat[2, 3]),
            )
            viewer._head_fwd_w = (
                float(-head_mat[0, 2]),
                float(-head_mat[1, 2]),
                float(-head_mat[2, 2]),
            )
        except Exception:
            pass

    def _initialize_screen_once(self, views):
        viewer = self.viewer
        if viewer._screen_eye_init or not views or views[0] is None:
            return
        try:
            if viewer._head_pos_w is not None:
                viewer._initial_head_y = float(viewer._head_pos_w[1])
        except Exception:
            pass
        viewer._reset_screen_to_default(show_border=False)
        viewer._screen_eye_init = True
