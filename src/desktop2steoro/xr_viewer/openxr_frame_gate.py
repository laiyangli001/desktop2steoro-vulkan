class OpenXRFrameGate:
    def __init__(self, viewer, frame_submitter):
        self.viewer = viewer
        self.frame_submitter = frame_submitter

    def handle_ready_or_stall(self, *, frame_state, now, composition_layers, submit_start):
        viewer = self.viewer
        session_idle_timeout = viewer._track_session_idle_render(frame_state.should_render, now)
        if frame_state.should_render:
            viewer._breakdown_inc('openxr_should_render')
        else:
            viewer._breakdown_inc('openxr_no_render')

        if viewer._session_ready_pending:
            if frame_state.should_render:
                viewer._session_ready_pending = False
                viewer._preview_only_mode = False
                viewer._waiting_for_headset = False
                viewer._resume_source_inference()
                viewer._set_render_active(True)
                viewer._source_resume_grace_until = now + viewer._source_resume_grace
                print("[OpenXRViewer] Headset detected, render confirmed")
                return False, session_idle_timeout
            self.submit_empty_frame(
                composition_layers=composition_layers,
                display_time=frame_state.predicted_display_time,
                submit_start=submit_start,
            )
            self.enter_idle_if_needed(session_idle_timeout)
            return True, session_idle_timeout

        if not viewer._has_fresh_source_frame(now):
            viewer._breakdown_inc('openxr_no_fresh')
            viewer._pause_xr_output_for_source_stall()
            if not viewer._has_renderable_source_frame():
                viewer._breakdown_inc('openxr_no_renderable')
                self.submit_empty_frame(
                    composition_layers=composition_layers,
                    display_time=frame_state.predicted_display_time,
                    submit_start=submit_start,
                )
                return True, session_idle_timeout
        return False, session_idle_timeout

    def submit_empty_frame(self, *, composition_layers, display_time, submit_start):
        if not composition_layers and not getattr(self.viewer, '_xr_swapchains', {}):
            self._prewarm_empty_frame_swapchain()
        self.frame_submitter.submit(
            composition_layers,
            display_time=display_time,
            submit_start=submit_start,
        )

    def _prewarm_empty_frame_swapchain(self):
        viewer = self.viewer
        ensure_projection = getattr(viewer, '_ensure_projection_swapchains', None)
        if callable(ensure_projection):
            try:
                return bool(ensure_projection())
            except Exception as exc:
                print(f"[OpenXRViewer] Projection empty-frame prewarm failed: {type(exc).__name__}: {exc}")
        return False

    def enter_idle_if_needed(self, session_idle_timeout):
        viewer = self.viewer
        if not session_idle_timeout or viewer._hard_idle_active:
            return False
        print(
            f"[OpenXRViewer] Session idle for {viewer._session_idle_render_timeout:.0f}s; "
            "source/render paused, keeping OpenXR session"
        )
        viewer._headset_wait_inference_deadline = 0.0
        viewer._headset_wait_inference_paused = True
        viewer._set_source_active(False)
        viewer._enter_hard_idle_wait()
        return True
