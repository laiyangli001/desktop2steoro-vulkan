# Desktop2Stereo OpenXR viewer: controller action creation and bindings.

try:
    import xr
except ImportError:
    xr = None


class CoreControllerActionsMixin:
    """OpenXR controller action setup and session binding."""

    def _attach_controller_actions_to_session(self):
        xr.attach_session_action_sets(
            self._xr_session,
            xr.SessionActionSetsAttachInfo(action_sets=[self._action_set]),
        )

        self._xr_actions_sync_info = xr.ActionsSyncInfo(active_action_sets=[
            xr.ActiveActionSet(
                action_set=self._action_set,
                subaction_path=xr.NULL_PATH,
            )
        ])

        # Create action spaces for aim poses (used to locate controller each frame)
        for act, attr in [
            (self._act_aim_left,  "_aim_space_l"),
            (self._act_aim_right, "_aim_space_r"),
        ]:
            try:
                space = xr.create_action_space(
                    self._xr_session,
                    xr.ActionSpaceCreateInfo(
                        action=act,
                        pose_in_action_space=xr.Posef(),
                    ),
                )
                setattr(self, attr, space)
            except Exception as e:
                print(f"[OpenXRViewer] Aim space creation failed: {e}")

        # Create action spaces for grip poses (used to place controller 3D models)
        for act, attr in [
            (self._act_grip_left,  "_grip_space_l"),
            (self._act_grip_right, "_grip_space_r"),
        ]:
            if act is None:
                continue
            try:
                space = xr.create_action_space(
                    self._xr_session,
                    xr.ActionSpaceCreateInfo(
                        action=act,
                        pose_in_action_space=xr.Posef(),
                    ),
                )
                setattr(self, attr, space)
            except Exception as e:
                print(f"[OpenXRViewer] Grip space creation failed: {e}")

    def _init_controller_actions(self):
        """Set up OpenXR action set with thumbstick and menu button actions."""
        self._action_set = xr.create_action_set(
            self._xr_instance,
            xr.ActionSetCreateInfo(
                action_set_name="screen_control",
                localized_action_set_name="Screen Control",
                priority=0,
            ),
        )
        subpaths = [
            xr.string_to_path(self._xr_instance, p)
            for p in ["/user/hand/left", "/user/hand/right"]
        ]
        # Cache hand XrPath values so per-frame action reads don't call string_to_path
        self._path_left  = subpaths[0]
        self._path_right = subpaths[1]

        def make_vec2(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.VECTOR2F_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_left_stick  = make_vec2("left_stick",  "Left Stick")
        self._act_right_stick = make_vec2("right_stick", "Right Stick")

        def make_bool(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_menu_btn  = make_bool("menu_btn",   "Menu Button")
        self._act_left_grip = make_bool("left_grip",  "Left Grip")
        self._act_right_grip= make_bool("right_grip", "Right Grip")
        self._act_a_btn     = make_bool("a_btn",      "A Button")
        self._act_b_btn     = make_bool("b_btn",      "B Button")
        self._act_x_btn     = make_bool("x_btn",      "X Button")
        self._act_y_btn     = make_bool("y_btn",      "Y Button")
        self._act_left_stick_click  = make_bool("left_stick_click",  "Left Stick Click")
        self._act_right_stick_click = make_bool("right_stick_click", "Right Stick Click")
        self._act_left_stick_touch  = make_bool("left_stick_touch",  "Left Stick Touch")
        self._act_right_stick_touch = make_bool("right_stick_touch", "Right Stick Touch")

        def make_float(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.FLOAT_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_left_trigger  = make_float("left_trigger",  "Left Trigger")
        self._act_right_trigger = make_float("right_trigger", "Right Trigger")

        self._act_haptic = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.VIBRATION_OUTPUT,
                action_name="haptic",
                localized_action_name="Controller Haptic",
                count_subaction_paths=len(subpaths),
                subaction_paths=subpaths,
            ),
        )

        self._act_aim_left = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="aim_left",
                localized_action_name="Left Aim Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[0]],
            ),
        )
        self._act_aim_right = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="aim_right",
                localized_action_name="Right Aim Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[1]],
            ),
        )

        # Grip pose actions used for placing controller 3D models
        self._act_grip_left = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="grip_left",
                localized_action_name="Left Grip Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[0]],
            ),
        )
        self._act_grip_right = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="grip_right",
                localized_action_name="Right Grip Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[1]],
            ),
        )

        # Per-profile binding table.
        # Use squeeze/value (float path) for grip -the runtime auto-thresholds it
        # for BOOLEAN_INPUT actions, and it works on more firmware versions than
        # squeeze/click (which requires a discrete click event on some runtimes).
        _b = {
            "/interaction_profiles/oculus/touch_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/thumbstick/touch",   self._act_left_stick_touch),
                ("/user/hand/right/input/thumbstick/touch",  self._act_right_stick_touch),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/x/click",            self._act_x_btn),
                ("/user/hand/left/input/y/click",            self._act_y_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
                ("/user/hand/left/output/haptic",            self._act_haptic),
                ("/user/hand/right/output/haptic",           self._act_haptic),
            ],
            "/interaction_profiles/valve/index_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/thumbstick/touch",   self._act_left_stick_touch),
                ("/user/hand/right/input/thumbstick/touch",  self._act_right_stick_touch),
                ("/user/hand/left/input/trackpad/touch",     self._act_left_stick_touch),
                ("/user/hand/right/input/trackpad/touch",    self._act_right_stick_touch),
                ("/user/hand/left/input/trackpad/click",     self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",         self._act_aim_right),
                ("/user/hand/left/input/grip/pose",         self._act_grip_left),
                ("/user/hand/right/input/grip/pose",        self._act_grip_right),
                ("/user/hand/left/output/haptic",           self._act_haptic),
                ("/user/hand/right/output/haptic",          self._act_haptic),
            ],
            # HTC Vive wand: trackpad (no thumbstick), squeeze/click (boolean,
            # no analog value), trigger value/click, menu, no A/B/X/Y buttons.
            # The trackpad's 2D parent binds to Vector2f stick actions, and
            # trackpad/click stands in for thumbstick click. Grip uses
            # squeeze/click directly since the wand has no analog squeeze.
            "/interaction_profiles/htc/vive_controller": [
                ("/user/hand/left/input/trackpad",           self._act_left_stick),
                ("/user/hand/right/input/trackpad",          self._act_right_stick),
                ("/user/hand/left/input/trackpad/click",     self._act_left_stick_click),
                ("/user/hand/right/input/trackpad/click",    self._act_right_stick_click),
                ("/user/hand/left/input/trackpad/touch",     self._act_left_stick_touch),
                ("/user/hand/right/input/trackpad/touch",    self._act_right_stick_touch),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/click",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/click",     self._act_right_grip),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
                ("/user/hand/left/output/haptic",            self._act_haptic),
                ("/user/hand/right/output/haptic",           self._act_haptic),
            ],
            # Windows Mixed Reality motion controllers expose a clickable trackpad.
            "/interaction_profiles/microsoft/motion_controller": [
                ("/user/hand/left/input/trackpad",           self._act_left_stick),
                ("/user/hand/right/input/trackpad",          self._act_right_stick),
                ("/user/hand/left/input/trackpad/click",     self._act_left_stick_click),
                ("/user/hand/right/input/trackpad/click",    self._act_right_stick_click),
                ("/user/hand/left/input/trackpad/touch",     self._act_left_stick_touch),
                ("/user/hand/right/input/trackpad/touch",    self._act_right_stick_touch),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/click",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/click",     self._act_right_grip),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
                ("/user/hand/left/output/haptic",            self._act_haptic),
                ("/user/hand/right/output/haptic",           self._act_haptic),
            ],
            # KHR simple only has select/click (boolean) and menu -no sticks or grip
            "/interaction_profiles/khr/simple_controller": [
                ("/user/hand/left/input/menu/click",    self._act_menu_btn),
                ("/user/hand/left/input/aim/pose",      self._act_aim_left),
                ("/user/hand/right/input/aim/pose",     self._act_aim_right),
                ("/user/hand/left/input/grip/pose",     self._act_grip_left),
                ("/user/hand/right/input/grip/pose",    self._act_grip_right),
                ("/user/hand/left/output/haptic",       self._act_haptic),
                ("/user/hand/right/output/haptic",      self._act_haptic),
            ],
            # PICO 4 Ultra controller interaction profile
            "/interaction_profiles/bytedance/pico_4u_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/thumbstick/touch",   self._act_left_stick_touch),
                ("/user/hand/right/input/thumbstick/touch",  self._act_right_stick_touch),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/x/click",            self._act_x_btn),
                ("/user/hand/left/input/y/click",            self._act_y_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
                ("/user/hand/left/output/haptic",            self._act_haptic),
                ("/user/hand/right/output/haptic",           self._act_haptic),
            ],
        }

        for profile, pairs in _b.items():
            try:
                xr.suggest_interaction_profile_bindings(
                    self._xr_instance,
                    xr.InteractionProfileSuggestedBinding(
                        interaction_profile=xr.string_to_path(self._xr_instance, profile),
                        suggested_bindings=[
                            xr.ActionSuggestedBinding(
                                action=act,
                                binding=xr.string_to_path(self._xr_instance, path),
                            )
                            for path, act in pairs
                        ],
                    ),
                )
            except Exception:
                pass

        self._attach_controller_actions_to_session()