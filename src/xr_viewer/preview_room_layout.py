#!/usr/bin/env python3
"""Preview a room profile's view_pose and screen layout without OpenXR."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
import time
from pathlib import Path

import glfw
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
ENVIRONMENTS_DIR = APP_DIR / "xr_viewer" / "environments"
PREVIEW_FINE_MOVE_SPEED_MPS = 1.0
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from xr_viewer.filament_preview_bridge import FilamentDesktopPreview  # noqa: E402


def _vec3(data, default):
    if isinstance(data, (list, tuple)) and len(data) >= 3:
        try:
            return [float(data[0]), float(data[1]), float(data[2])]
        except (TypeError, ValueError):
            pass
    return list(default)


def _rot_deg(data, default=(0.0, 0.0, 0.0)):
    return [math.radians(v) for v in _vec3(data, default)]


def _active_view_pose(profile: dict) -> dict:
    view_poses = profile.get("view_poses")
    if isinstance(view_poses, list) and view_poses:
        try:
            idx = int(profile.get("view_pose_index", 0)) % len(view_poses)
        except (TypeError, ValueError):
            idx = 0
        if isinstance(view_poses[idx], dict):
            return view_poses[idx]
    view = profile.get("view_pose", profile.get("camera", {}))
    return view if isinstance(view, dict) else {}


def _pose_position(view: dict, default):
    if isinstance(view, dict):
        if "position" in view:
            return _vec3(view.get("position"), default)
        if all(key in view for key in ("x", "y", "z")):
            return _vec3([view.get("x"), view.get("y"), view.get("z")], default)
    return list(default)


def _pose_rotation_deg(view: dict, default=(0.0, 0.0, 0.0)):
    if isinstance(view, dict):
        if "rotation_deg" in view:
            return _vec3(view.get("rotation_deg"), default)
        if "rotation" in view:
            return [math.degrees(v) for v in _rot_deg(view.get("rotation"), default)]
        if "angle" in view:
            return [float(view.get("angle") or 0.0), 0.0, 0.0]
    return list(default)


def _set_pose_position(view: dict, pos):
    rounded = [round(float(v), 4) for v in pos]
    if any(key in view for key in ("x", "y", "z")):
        view["x"], view["y"], view["z"] = rounded
    else:
        view["position"] = rounded


def _set_pose_rotation_deg(view: dict, rot):
    rounded = [round(float(v), 3) for v in rot]
    view["rotation_deg"] = rounded
    if "angle" in view:
        view["angle"] = rounded[0]


def _resolve_room_dir(room: str) -> Path:
    room_dir = ENVIRONMENTS_DIR / room
    if room_dir.exists():
        return room_dir
    room_key = room.strip().lower()
    if ENVIRONMENTS_DIR.exists():
        for candidate in ENVIRONMENTS_DIR.iterdir():
            if candidate.is_dir() and candidate.name.lower() == room_key:
                return candidate
    return room_dir


def _native_window_handle(window) -> int:
    """Return the platform window handle accepted by Filament's SwapChain."""
    if sys.platform == "win32":
        return int(glfw.get_win32_window(window))
    if sys.platform == "linux":
        return int(glfw.get_x11_window(window))
    if sys.platform == "darwin":
        return int(glfw.get_cocoa_window(window))
    raise RuntimeError(f"Unsupported desktop platform: {sys.platform}")


def _load_profile(room: str):
    room_dir = _resolve_room_dir(room)
    profile_path = room_dir / "profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"profile.json not found: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    if not isinstance(profile, dict):
        raise ValueError(f"profile.json root must be object: {profile_path}")

    glb_name = str(profile.get("glb", "environment.glb") or "environment.glb")
    glb_path = Path(glb_name)
    if not glb_path.is_absolute():
        glb_path = room_dir / glb_name
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found: {glb_path}")
    return room_dir, profile_path, profile, glb_path


def _save_profile(path: Path, profile: dict):
    # Runtime reads GLB-embedded KHR_lights_punctual lights, not profile.gltf_lights.
    # Keep saved room profiles aligned with xrviewer_env.py's profile schema.
    profile.pop("gltf_lights", None)
    profile.setdefault("env_fill_lights", [])
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _mat_from_trs(pos, rot_rad, scale=(1.0, 1.0, 1.0)):
    yaw, pitch, roll = rot_rad
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype="f4")
    rx = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype="f4")
    rz = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype="f4")
    sm = np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0]).astype("f4")
    tm = np.eye(4, dtype="f4")
    tm[:3, 3] = np.array(pos, dtype="f4")
    return tm @ ry @ rx @ rz @ sm


def _profile_model_matrix(profile):
    model_pos = _vec3(profile.get("model_position"), [0.0, 0.0, 0.0])
    model_rot = _rot_deg(
        profile.get("model_rotation_deg", profile.get("model_rotation")),
        [0.0, 0.0, 0.0],
    )
    model_scale = _vec3(profile.get("model_scale"), [1.0, 1.0, 1.0])
    return _mat_from_trs(model_pos, model_rot, model_scale)


def _pose_position_in_scene(profile, view):
    world_position = np.asarray(_pose_position(view, [0.0, 1.2, 0.0]), dtype="f4")
    scene_position = np.linalg.inv(_profile_model_matrix(profile)) @ np.append(world_position, 1.0)
    return scene_position[:3].astype("f4").tolist()


def _scene_position_in_profile(profile, position):
    scene_position = np.asarray(position, dtype="f4")
    world_position = _profile_model_matrix(profile) @ np.append(scene_position, 1.0)
    return world_position[:3].astype("f4").tolist()


def _profile_projection_planes(profile):
    try:
        near = max(0.01, float(profile.get("xr_projection_near", 0.03)))
    except (TypeError, ValueError):
        near = 0.03
    try:
        far = max(near + 1.0, float(profile.get("xr_projection_far", 200.0)))
    except (TypeError, ValueError):
        far = 200.0
    return near, far


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("room", nargs="?", default="bedroom")
    parser.add_argument(
        "--max-texture-size",
        type=int,
        default=0,
        help="Maximum embedded texture edge for desktop preview; 0 keeps source resolution",
    )
    parser.add_argument("--exposure", type=float, default=None, help="Filament exposure in EV")
    parser.add_argument("--skybox-brightness", type=float, default=None, help="Skybox brightness multiplier")
    parser.add_argument("--fill-light-intensity", type=float, default=None, help="Filament directional fill light intensity")
    parser.add_argument("--center-view", action="store_true", help="Start camera at the transformed model bounds center")
    args = parser.parse_args()

    os.chdir(APP_DIR)
    room_dir, profile_path, profile, glb_path = _load_profile(args.room)
    projection_near, projection_far = _profile_projection_planes(profile)
    view_pose = _active_view_pose(profile)
    if not view_pose:
        view_pose = profile.setdefault("view_pose", {})
    screen = profile.setdefault("screen", {})
    screen.setdefault("name", "Preview Screen")
    screen.setdefault("width", 2.4)
    screen.setdefault("position", [0.0, 1.2, -2.0])
    screen.setdefault("rotation_deg", [0.0, 0.0, 0.0])

    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(1280, 720, f"Room Layout Preview - {args.room}", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    native_window = _native_window_handle(window)
    preview = FilamentDesktopPreview(native_window, 1280, 720)
    preview.load_glb(glb_path.read_bytes(), max_texture_dimension=args.max_texture_size)
    preview_exposure = float(
        args.exposure if args.exposure is not None else profile.get("preview_exposure", 2.0)
    )
    skybox_brightness = float(
        args.skybox_brightness
        if args.skybox_brightness is not None
        else profile.get("preview_skybox_brightness", 1.0)
    )
    fill_light_color = _vec3(profile.get("preview_fill_light_color"), [1.0, 0.88, 0.78])
    fill_light_direction = _vec3(profile.get("preview_fill_light_direction"), [-0.35, -1.0, -0.55])
    fill_light_intensity = float(
        args.fill_light_intensity
        if args.fill_light_intensity is not None
        else profile.get("preview_fill_light_intensity", 100000.0)
    )
    preview.set_exposure(preview_exposure)
    preview.set_fill_light(fill_light_color, fill_light_intensity, fill_light_direction)
    preview.set_skybox_brightness(skybox_brightness)

    # Profile positions are stored in world coordinates; Filament renders the raw GLB scene.
    view_pos = _pose_position_in_scene(profile, view_pose)
    if args.center_view:
        print("--center-view is ignored by the Filament preview; use VIEW controls to adjust the profile seat.")
    view_rot_deg = _pose_rotation_deg(view_pose, [0.0, 0.0, 0.0])
    view_rot = [math.radians(v) for v in view_rot_deg]
    speed, size_speed = 1.0, 0.8
    rot_speed = 45.0
    saved_flash = 0.0
    exposure_key_cooldown = 0.0
    skybox_key_cooldown = 0.0
    edit_target = "SCREEN"
    tab_was_down = False
    mouse_look = False
    last_mouse = (0.0, 0.0)

    print(f"Room: {args.room}")
    print(f"Profile: {profile_path}")
    print(f"Preview projection: clip={projection_near:.3f}/{projection_far:.1f}")
    print("Preview animations: all embedded GLB animations enabled")
    print(f"Preview color: exposure={preview_exposure:.2f}EV skybox={skybox_brightness:.2f} fill={fill_light_intensity:.0f}")
    print(f"Preview navigation: move_speed={speed:.2f}m/s size_speed={size_speed:.2f}m/s")
    print(f"Preview fine mode: hold Ctrl for {PREVIEW_FINE_MOVE_SPEED_MPS:.2f}m/s movement/size adjustment")
    print("Controls:")
    print("  Tab: switch edit target SCREEN/VIEW")
    print("  SCREEN: Arrow=screen X/Y, PageUp/PageDown=screen Z, +/-=width")
    print("  GLOBAL: [ / ]=seat exposure down/up, , / .=skybox brightness down/up")
    print("  SCREEN: 1=27in monitor, 2=65in TV, 3=100in projector, 4=cinema")
    print("  VIEW:   A/D=seat X, Up/Down or Space/LeftShift=seat Y, W/S=seat Z")
    print("  Mouse:  hold right button and drag to rotate VIEW yaw/pitch")
    print("  Both:   Q/E=yaw, T/G=pitch, Z/C=roll")
    print("  P: save profile, R: reload profile, Esc: exit")

    def mouse_button_cb(_window, button, action, _mods):
        nonlocal mouse_look, last_mouse
        if button == glfw.MOUSE_BUTTON_RIGHT:
            mouse_look = action == glfw.PRESS
            last_mouse = glfw.get_cursor_pos(window)

    def cursor_pos_cb(_window, x, y):
        nonlocal last_mouse, view_rot, view_pose
        if not mouse_look:
            last_mouse = (x, y)
            return
        dx = x - last_mouse[0]
        dy = y - last_mouse[1]
        last_mouse = (x, y)
        view_rot_deg = _pose_rotation_deg(view_pose, [math.degrees(v) for v in view_rot])
        view_rot_deg[0] -= dx * 0.12
        view_rot_deg[1] = max(-89.0, min(89.0, view_rot_deg[1] - dy * 0.12))
        _set_pose_rotation_deg(view_pose, view_rot_deg)
        view_rot = [math.radians(v) for v in view_rot_deg]

    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)

    def key_down(key):
        return glfw.get_key(window, key) in (glfw.PRESS, glfw.REPEAT)

    def ctrl_down():
        return key_down(glfw.KEY_LEFT_CONTROL) or key_down(glfw.KEY_RIGHT_CONTROL)

    last_time = glfw.get_time()
    animation_start_time = last_time
    next_frame_time = last_time
    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = max(0.001, min(0.05, now - last_time))
        last_time = now
        glfw.poll_events()
        exposure_key_cooldown = max(0.0, exposure_key_cooldown - dt)
        skybox_key_cooldown = max(0.0, skybox_key_cooldown - dt)
        if exposure_key_cooldown <= 0.0:
            if key_down(glfw.KEY_LEFT_BRACKET):
                preview_exposure = max(-8.0, preview_exposure - 0.25)
                preview.set_exposure(preview_exposure)
                exposure_key_cooldown = 0.12
            elif key_down(glfw.KEY_RIGHT_BRACKET):
                preview_exposure = min(8.0, preview_exposure + 0.25)
                preview.set_exposure(preview_exposure)
                exposure_key_cooldown = 0.12
        if skybox_key_cooldown <= 0.0:
            if key_down(glfw.KEY_COMMA):
                skybox_brightness = max(0.0, skybox_brightness - 0.05)
                preview.set_skybox_brightness(skybox_brightness)
                skybox_key_cooldown = 0.12
            elif key_down(glfw.KEY_PERIOD):
                skybox_brightness = min(16.0, skybox_brightness + 0.05)
                preview.set_skybox_brightness(skybox_brightness)
                skybox_key_cooldown = 0.12

        tab_down = glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS
        if tab_down and not tab_was_down:
            edit_target = "VIEW" if edit_target == "SCREEN" else "SCREEN"
        tab_was_down = tab_down

        pos = _vec3(screen.get("position"), [0.0, 1.2, -2.0])
        rot = _vec3(screen.get("rotation_deg"), [0.0, 0.0, 0.0])
        view_rot_deg = _pose_rotation_deg(view_pose, [math.degrees(v) for v in view_rot])
        changed_screen = False
        changed_view = False

        fine_mode = ctrl_down()
        active_move_speed = PREVIEW_FINE_MOVE_SPEED_MPS if fine_mode else speed
        active_size_speed = PREVIEW_FINE_MOVE_SPEED_MPS if fine_mode else size_speed
        step = active_move_speed * dt
        rstep = rot_speed * dt

        if edit_target == "SCREEN":
            size_presets = {
                glfw.KEY_1: ("Desk Monitor", 0.62),
                glfw.KEY_2: ("65in TV", 1.44),
                glfw.KEY_3: ("Default Projector", 2.4),
                glfw.KEY_4: ("Cinema Screen", 8.0),
            }
            for preset_key, (preset_name, preset_width) in size_presets.items():
                if key_down(preset_key):
                    screen["name"] = preset_name
                    screen["width"] = preset_width
                    changed_screen = True
            if key_down(glfw.KEY_LEFT):
                pos[0] -= step; changed_screen = True
            if key_down(glfw.KEY_RIGHT):
                pos[0] += step; changed_screen = True
            if key_down(glfw.KEY_UP):
                pos[1] += step; changed_screen = True
            if key_down(glfw.KEY_DOWN):
                pos[1] -= step; changed_screen = True
            if key_down(glfw.KEY_PAGE_UP):
                pos[2] += step; changed_screen = True
            if key_down(glfw.KEY_PAGE_DOWN):
                pos[2] -= step; changed_screen = True
            if key_down(glfw.KEY_EQUAL) or key_down(glfw.KEY_KP_ADD):
                screen["width"] = round(max(0.05, float(screen.get("width", 2.4)) + active_size_speed * dt), 4)
                changed_screen = True
            if key_down(glfw.KEY_MINUS) or key_down(glfw.KEY_KP_SUBTRACT):
                screen["width"] = round(max(0.05, float(screen.get("width", 2.4)) - active_size_speed * dt), 4)
                changed_screen = True
            if key_down(glfw.KEY_Q):
                rot[0] += rstep; changed_screen = True
            if key_down(glfw.KEY_E):
                rot[0] -= rstep; changed_screen = True
            if key_down(glfw.KEY_T):
                rot[1] += rstep; changed_screen = True
            if key_down(glfw.KEY_G):
                rot[1] -= rstep; changed_screen = True
            if key_down(glfw.KEY_Z):
                rot[2] += rstep; changed_screen = True
            if key_down(glfw.KEY_C):
                rot[2] -= rstep; changed_screen = True
        else:
            yaw_rad = math.radians(view_rot_deg[0])
            forward = np.array([-math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype="f4")
            right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)], dtype="f4")
            if key_down(glfw.KEY_W):
                view_pos = (np.array(view_pos) + forward * step).tolist(); changed_view = True
            if key_down(glfw.KEY_S):
                view_pos = (np.array(view_pos) - forward * step).tolist(); changed_view = True
            if key_down(glfw.KEY_A):
                view_pos = (np.array(view_pos) - right * step).tolist(); changed_view = True
            if key_down(glfw.KEY_D):
                view_pos = (np.array(view_pos) + right * step).tolist(); changed_view = True
            if key_down(glfw.KEY_SPACE) or key_down(glfw.KEY_UP):
                view_pos[1] += step; changed_view = True
            if key_down(glfw.KEY_LEFT_SHIFT) or key_down(glfw.KEY_RIGHT_SHIFT) or key_down(glfw.KEY_DOWN):
                view_pos[1] -= step; changed_view = True
            if key_down(glfw.KEY_Q):
                view_rot_deg[0] += rstep; changed_view = True
            if key_down(glfw.KEY_E):
                view_rot_deg[0] -= rstep; changed_view = True
            if key_down(glfw.KEY_T):
                view_rot_deg[1] += rstep; changed_view = True
            if key_down(glfw.KEY_G):
                view_rot_deg[1] -= rstep; changed_view = True
            if key_down(glfw.KEY_Z):
                view_rot_deg[2] += rstep; changed_view = True
            if key_down(glfw.KEY_C):
                view_rot_deg[2] -= rstep; changed_view = True

        if changed_screen:
            screen["position"] = [round(v, 4) for v in pos]
            screen["rotation_deg"] = [round(v, 3) for v in rot]
        if changed_view:
            _set_pose_position(view_pose, _scene_position_in_profile(profile, view_pos))
            _set_pose_rotation_deg(view_pose, view_rot_deg)
            view_rot = [math.radians(v) for v in view_rot_deg]

        if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS:
            _save_profile(profile_path, profile)
            saved_flash = 1.0
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            _room_dir, _profile_path, profile, _glb_path = _load_profile(args.room)
            projection_near, projection_far = _profile_projection_planes(profile)
            view_pose = _active_view_pose(profile)
            if not view_pose:
                view_pose = profile.setdefault("view_pose", {})
            screen = profile.setdefault("screen", {})
            view_pos = _pose_position_in_scene(profile, view_pose)
            view_rot_deg = _pose_rotation_deg(view_pose, [0.0, 0.0, 0.0])
            view_rot = [math.radians(v) for v in view_rot_deg]
            animation_start_time = glfw.get_time()
            speed, size_speed = 1.0, 0.8
            skybox_brightness = float(profile.get("preview_skybox_brightness", 1.0))
            preview.set_skybox_brightness(skybox_brightness)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        title = (
            f"{args.room} | {edit_target} | {screen.get('name', 'Screen')} | "
            f"exposure={preview_exposure:.2f}EV skybox={skybox_brightness:.2f} | view={view_pos} {view_rot_deg} | "
            f"pos={screen.get('position')} rot={screen.get('rotation_deg')} "
            f"w={float(screen.get('width', 2.4)):.3f}m"
        )
        if saved_flash > 0:
            title += " | SAVED"
            saved_flash -= dt
        glfw.set_window_title(window, title)

        ww, wh = glfw.get_window_size(window)
        if ww <= 0 or wh <= 0:
            glfw.poll_events()
            continue
        aspect = ww / wh
        if (ww, wh) != getattr(preview, "viewport_size", None):
            preview.set_viewport(ww, wh)
            preview.viewport_size = (ww, wh)
        yaw, pitch, roll = view_rot
        rotation = _mat_from_trs(view_pos, (yaw, pitch, roll))[:3, :3]
        forward = rotation @ np.array([0.0, 0.0, -1.0], dtype="f4")
        up = rotation @ np.array([0.0, 1.0, 0.0], dtype="f4")
        center = np.asarray(view_pos, dtype="f4") + forward
        preview.set_camera(view_pos, center.tolist(), up.tolist())
        preview.set_projection(80.0, aspect, projection_near, projection_far)
        animation_time = max(0.0, now - animation_start_time)
        preview.apply_animations(animation_time)
        preview.render()
        next_frame_time += 1.0 / 60.0
        delay = next_frame_time - glfw.get_time()
        if delay > 0.0:
            time.sleep(min(delay, 0.02))
        else:
            next_frame_time = glfw.get_time()

    preview.close()
    glfw.terminate()


if __name__ == "__main__":
    main()
