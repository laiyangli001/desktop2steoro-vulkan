import time

try:
    import xr
except ImportError:
    xr = None


class OpenXRFrameTiming:
    def __init__(self, viewer):
        self.viewer = viewer

    def _log_predicted_pacing(self, predicted_delta_s):
        return

    def begin_frame(self, *, breakdown_enabled=False):
        viewer = self.viewer
        wait_start = time.perf_counter() if breakdown_enabled else 0.0
        try:
            frame_state = xr.wait_frame(viewer._xr_session, viewer._xr_frame_wait_info)
        except Exception as exc:
            print(f"[OpenXRViewer] xr.wait_frame failed: {type(exc).__name__}: {exc}")
            raise
        if breakdown_enabled:
            viewer._breakdown_add_time('openxr_wait_frame', time.perf_counter() - wait_start)
            predicted_time = getattr(frame_state, 'predicted_display_time', None)
            previous_time = getattr(viewer, '_last_xr_predicted_display_time', None)
            viewer._last_xr_predicted_display_time = predicted_time
            if predicted_time is not None and previous_time is not None:
                predicted_delta_s = (int(predicted_time) - int(previous_time)) / 1_000_000_000.0
                if 0.0 < predicted_delta_s < 1.0:
                    viewer._breakdown_add_time('openxr_predicted_period', predicted_delta_s)
                    self._log_predicted_pacing(predicted_delta_s)
        submit_start = time.perf_counter() if breakdown_enabled else 0.0
        try:
            xr.begin_frame(viewer._xr_session, viewer._xr_frame_begin_info)
        except Exception as exc:
            print(f"[OpenXRViewer] xr.begin_frame failed: {type(exc).__name__}: {exc}")
            raise
        return frame_state, submit_start
