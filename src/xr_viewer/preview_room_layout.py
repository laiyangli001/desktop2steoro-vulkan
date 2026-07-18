#!/usr/bin/env python3
"""Preview a room profile's view_pose and screen layout without OpenXR."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import glfw
import moderngl
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

from xr_viewer.gl_state import set_depth_mask  # noqa: E402
from xr_viewer.gltf import (  # noqa: E402
    OPENGL_VERTEX_FORMAT,
    apply_skybox_profile,
    format_gltf_scene_summary,
    load_glb_model,
    render_pass_from_primitive,
    sort_transparent_primitives,
    summarize_gltf_scene,
    validate_mesh_contract,
)


ENV_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec2 in_uv1;
out vec3 v_normal;
out vec3 v_position;
out vec2 v_uv;
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform int u_base_texcoord;
void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_uv = u_base_texcoord == 1 ? in_uv1 : in_uv;
    gl_Position = u_mvp * world_pos;
}
"""

ENV_FRAG = """
#version 330
in vec3 v_normal;
in vec3 v_position;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform int u_use_texture;
uniform vec3 u_base_color;
uniform vec3 u_camera_pos;
uniform vec3 u_ambient_color;
uniform vec3 u_light_color;
uniform float u_alpha;
uniform int u_alpha_mode;
uniform float u_alpha_cutoff;
uniform float u_exposure;
uniform float u_gamma;

vec3 gltfSrgbToLinear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 gltfToneMap(vec3 linearColor) {
    linearColor = max(linearColor, vec3(0.0));
    return linearColor / (linearColor + vec3(1.0));
}

vec3 gltfLinearToOutput(vec3 linearColor, float gamma) {
    return pow(clamp(gltfToneMap(linearColor), 0.0, 1.0), vec3(1.0 / max(gamma, 0.001)));
}

void main() {
    vec3 base = u_base_color;
    float alpha = u_alpha;
    if (u_use_texture == 1) {
        vec4 texel = texture(u_tex, v_uv);
        base *= gltfSrgbToLinear(texel.rgb);
        if (u_alpha_mode != 0) {
            alpha *= texel.a;
        }
    }
    if (u_alpha_mode == 1 && alpha < u_alpha_cutoff) {
        discard;
    }
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_camera_pos + vec3(0.0, 0.2, 0.0) - v_position);
    float diff = max(abs(dot(N, L)), 0.12);
    vec3 color = base * (u_ambient_color + u_light_color * diff) * u_exposure;
    fragColor = vec4(gltfLinearToOutput(color, u_gamma), alpha);
}
"""

SCREEN_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
uniform mat4 u_mvp;
void main() {
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

SCREEN_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
uniform vec4 u_color;
void main() {
    vec2 g = abs(fract(v_uv * vec2(16.0, 9.0)) - 0.5);
    float line = step(0.47, max(g.x, g.y));
    vec3 grid = mix(u_color.rgb, vec3(1.0), line * 0.35);
    fragColor = vec4(grid, u_color.a);
}
"""


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


def _view_matrix(pos, rot_rad):
    yaw, pitch, roll = rot_rad
    model = _mat_from_trs(pos, (yaw, pitch, roll), (1.0, 1.0, 1.0))
    return np.linalg.inv(model).astype("f4")


def _environment_model_matrix(profile):
    model_pos = _vec3(profile.get("model_position"), [0.0, -1.0, -3.0])
    model_rot = _rot_deg(profile.get("model_rotation_deg", profile.get("model_rotation")), [0.0, 0.0, 0.0])
    model_scale = _vec3(profile.get("model_scale"), [1.0, 1.0, 1.0])
    return _mat_from_trs(model_pos, model_rot, model_scale)


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


def _projection(aspect, fov_deg=80.0, near=0.03, far=200.0):
    try:
        aspect = float(aspect)
    except (TypeError, ValueError):
        aspect = 1.0
    if not math.isfinite(aspect) or aspect <= 0.0:
        aspect = 1.0
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ], dtype="f4")


def _screen_vertices(screen):
    width = float(screen.get("width", 2.4))
    height = float(screen.get("height", width * 9.0 / 16.0))
    pos = _vec3(screen.get("position"), [0.0, 1.2, -2.0])
    rot = _rot_deg(screen.get("rotation_deg", screen.get("rotation")), [0.0, 0.0, 0.0])
    model = _mat_from_trs(pos, rot, (1.0, 1.0, 1.0))
    corners = np.array([
        [-width / 2, -height / 2, 0, 0, 0],
        [ width / 2, -height / 2, 0, 1, 0],
        [-width / 2,  height / 2, 0, 0, 1],
        [ width / 2,  height / 2, 0, 1, 1],
    ], dtype="f4")
    p = np.c_[corners[:, :3], np.ones(4, dtype="f4")]
    corners[:, :3] = (model @ p.T).T[:, :3]
    return corners


def _make_env_resources(ctx, prog, glb_path: Path, profile):
    prims_data, textures, lights = load_glb_model(str(glb_path))
    apply_skybox_profile(prims_data, profile)
    summary = summarize_gltf_scene(prims_data, textures, lights)
    print("[Preview] " + format_gltf_scene_summary(summary, label=f"Active environment {glb_path}"))
    skybox = profile.get("skybox", {})
    skybox_mipmaps = bool(skybox.get("mipmaps", False)) if isinstance(skybox, dict) else False
    skybox_tex_ids = {
        int(pd.get("tex_id", -1))
        for pd in prims_data
        if pd.get("render_pass") == "sky"
    }
    local_min = None
    local_max = None
    tex_cache = {}
    for tid, arr in enumerate(textures):
        if arr is None:
            continue
        h, w = arr.shape[:2]
        tex = ctx.texture((w, h), 4, arr.tobytes())
        if tid in skybox_tex_ids and not skybox_mipmaps:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.build_mipmaps()
            tex.anisotropy = 8.0
        tex_cache[tid] = tex

    prims = []
    for pd in prims_data:
        validate_mesh_contract(pd["vertices"], pd["tangent"], pd["indices"])
        vertices = pd["vertices"].astype("f4", copy=False)
        if vertices.size:
            pos = vertices[:, :3]
            mn = pos.min(axis=0)
            mx = pos.max(axis=0)
            local_min = mn if local_min is None else np.minimum(local_min, mn)
            local_max = mx if local_max is None else np.maximum(local_max, mx)
        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(pd["indices"].astype("u4").tobytes())
        vao = ctx.vertex_array(
            prog,
            [(vbo, OPENGL_VERTEX_FORMAT, "in_position", "in_normal", "in_uv", "in_uv1")],
            ibo,
        )
        prims.append({
            "vao": vao,
            "tex_id": int(pd.get("tex_id", -1)),
            "base_color": np.array(pd.get("base_color", [1.0, 1.0, 1.0]), dtype="f4"),
            "base_alpha": float(pd.get("base_alpha", 1.0)),
            "alpha_mode": str(pd.get("alpha_mode", "OPAQUE") or "OPAQUE").upper(),
            "alpha_cutoff": float(pd.get("alpha_cutoff", 0.5)),
            "base_texcoord": int(pd.get("base_texcoord", 0) or 0),
            "render_pass": render_pass_from_primitive(pd),
            "sort_center_local": (
                vertices[:, :3].mean(axis=0).astype("f4")
                if vertices.size
                else np.zeros(3, dtype="f4")
            ),
        })
    return prims, tex_cache, local_min, local_max


def _world_bounds_from_local(local_min, local_max, model):
    if local_min is None or local_max is None:
        return None, None
    corners = np.array([
        [x, y, z, 1.0]
        for x in (float(local_min[0]), float(local_max[0]))
        for y in (float(local_min[1]), float(local_max[1]))
        for z in (float(local_min[2]), float(local_max[2]))
    ], dtype="f4")
    world = (model @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _preview_motion_speeds(env_world_min, env_world_max):
    base_move_speed = 0.75
    base_size_speed = 0.8
    if env_world_min is None or env_world_max is None:
        return base_move_speed, base_size_speed

    bounds_size = np.asarray(env_world_max, dtype=np.float64) - np.asarray(env_world_min, dtype=np.float64)
    if bounds_size.size == 0:
        return base_move_speed, base_size_speed

    max_extent = float(np.nanmax(np.abs(bounds_size)))
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        return base_move_speed, base_size_speed

    scene_scale = max(1.0, min(80.0, max_extent / 50.0))
    return base_move_speed * scene_scale, base_size_speed * scene_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("room", nargs="?", default="bedroom")
    parser.add_argument("--exposure", type=float, default=None, help="Preview-only brightness multiplier")
    parser.add_argument("--gamma", type=float, default=None, help="Preview-only output gamma")
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
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(1280, 720, f"Room Layout Preview - {args.room}", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    env_prog = ctx.program(vertex_shader=ENV_VERT, fragment_shader=ENV_FRAG)
    screen_prog = ctx.program(vertex_shader=SCREEN_VERT, fragment_shader=SCREEN_FRAG)

    env_prims, tex_cache, env_local_min, env_local_max = _make_env_resources(ctx, env_prog, glb_path, profile)
    screen_vbo = ctx.buffer(reserve=4 * 5 * 4)
    screen_vao = ctx.vertex_array(screen_prog, [(screen_vbo, "3f 2f", "in_position", "in_uv")])

    env_model = _environment_model_matrix(profile)

    view_pos = _pose_position(view_pose, [0.0, 1.2, 0.0])
    env_world_min, env_world_max = _world_bounds_from_local(env_local_min, env_local_max, env_model)
    if args.center_view and env_world_min is not None and env_world_max is not None:
        view_pos = ((env_world_min + env_world_max) * 0.5).astype(float).tolist()
        _set_pose_position(view_pose, view_pos)
    view_rot_deg = _pose_rotation_deg(view_pose, [0.0, 0.0, 0.0])
    view_rot = [math.radians(v) for v in view_rot_deg]
    preview_exposure = float(args.exposure if args.exposure is not None else profile.get("preview_exposure", 2.2))
    preview_gamma = float(args.gamma if args.gamma is not None else profile.get("preview_gamma", 2.2))
    speed, size_speed = _preview_motion_speeds(env_world_min, env_world_max)
    rot_speed = 45.0
    saved_flash = 0.0
    edit_target = "SCREEN"
    tab_was_down = False
    mouse_look = False
    last_mouse = (0.0, 0.0)

    print(f"Room: {args.room}")
    print(f"Profile: {profile_path}")
    print(f"Preview lighting: exposure={preview_exposure:.2f} gamma={preview_gamma:.2f}")
    print(f"Preview projection: clip={projection_near:.3f}/{projection_far:.1f}")
    print(f"Preview navigation: move_speed={speed:.2f}m/s size_speed={size_speed:.2f}m/s")
    print(f"Preview fine mode: hold Ctrl for {PREVIEW_FINE_MOVE_SPEED_MPS:.2f}m/s movement/size adjustment")
    print("Controls:")
    print("  Tab: switch edit target SCREEN/VIEW")
    print("  SCREEN: Arrow=screen X/Y, PageUp/PageDown=screen Z, +/-=width")
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
    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = max(0.001, min(0.05, now - last_time))
        last_time = now
        glfw.poll_events()

        tab_down = glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS
        if tab_down and not tab_was_down:
            edit_target = "VIEW" if edit_target == "SCREEN" else "SCREEN"
        tab_was_down = tab_down

        pos = _vec3(screen.get("position"), [0.0, 1.2, -2.0])
        rot = _vec3(screen.get("rotation_deg"), [0.0, 0.0, 0.0])
        view_pos = _pose_position(view_pose, view_pos)
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
            _set_pose_position(view_pose, view_pos)
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
            view_pos = _pose_position(view_pose, [0.0, 1.2, 0.0])
            view_rot_deg = _pose_rotation_deg(view_pose, [0.0, 0.0, 0.0])
            view_rot = [math.radians(v) for v in view_rot_deg]
            env_model = _environment_model_matrix(profile)
            env_world_min, env_world_max = _world_bounds_from_local(env_local_min, env_local_max, env_model)
            speed, size_speed = _preview_motion_speeds(env_world_min, env_world_max)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        title = (
            f"{args.room} | {edit_target} | {screen.get('name', 'Screen')} | "
            f"view={view_pos} {view_rot_deg} | "
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
        ctx.viewport = (0, 0, ww, wh)
        aspect = ww / wh
        proj = _projection(aspect, near=projection_near, far=projection_far)
        view = _view_matrix(view_pos, view_rot)
        vp = proj @ view
        cam_pos = np.array(view_pos, dtype="f4")

        ctx.clear(1.0, 1.0, 1.0, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)

        env_prog["u_mvp"].write(vp.T.astype("f4").tobytes())
        env_prog["u_model"].write(env_model.T.astype("f4").tobytes())
        env_prog["u_camera_pos"].write(cam_pos.tobytes())
        ambient = np.maximum(np.array(_vec3(profile.get("env_ambient_color"), [0.24, 0.24, 0.26]), dtype="f4"), 0.22)
        light = np.maximum(np.array(_vec3(profile.get("env_head_light_color"), [0.70, 0.70, 0.72]), dtype="f4"), 0.85)
        env_prog["u_ambient_color"].value = (float(ambient[0]), float(ambient[1]), float(ambient[2]))
        env_prog["u_light_color"].value = (float(light[0]), float(light[1]), float(light[2]))
        env_prog["u_exposure"].value = max(0.05, preview_exposure)
        env_prog["u_gamma"].value = max(0.1, preview_gamma)
        def draw_env_prim(prim):
            tid = prim["tex_id"]
            if tid in tex_cache:
                tex_cache[tid].use(location=0)
                env_prog["u_use_texture"].value = 1
            else:
                env_prog["u_use_texture"].value = 0
            bc = prim["base_color"]
            alpha_mode = "OPAQUE" if prim.get("render_pass") == "sky" else prim.get("alpha_mode", "OPAQUE")
            alpha_mode_id = 1 if alpha_mode == "MASK" else (2 if alpha_mode == "BLEND" else 0)
            env_prog["u_base_texcoord"].value = 1 if int(prim.get("base_texcoord", 0) or 0) == 1 else 0
            env_prog["u_base_color"].value = (float(bc[0]), float(bc[1]), float(bc[2]))
            env_prog["u_alpha"].value = min(max(float(prim["base_alpha"]), 0.0), 1.0)
            env_prog["u_alpha_mode"].value = alpha_mode_id
            env_prog["u_alpha_cutoff"].value = float(prim.get("alpha_cutoff", 0.5))
            prim["vao"].render(moderngl.TRIANGLES)

        sky_prims = [prim for prim in env_prims if prim.get("render_pass") == "sky"]
        solid_prims = [
            prim for prim in env_prims
            if prim.get("render_pass") in ("opaque", "mask")
        ]
        transparent_prims = [
            prim for prim in env_prims if prim.get("render_pass") == "transparent"
        ]
        if sky_prims:
            ctx.disable(moderngl.CULL_FACE)
            set_depth_mask(False)
            for prim in sky_prims:
                draw_env_prim(prim)
            set_depth_mask(True)
        for prim in solid_prims:
            draw_env_prim(prim)
        if transparent_prims:
            transparent_prims = sort_transparent_primitives(
                transparent_prims,
                cam_pos,
                env_model,
            )
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            set_depth_mask(False)
            for prim in transparent_prims:
                draw_env_prim(prim)
            set_depth_mask(True)
            ctx.disable(moderngl.BLEND)

        # Render the configured screen as a translucent blue grid.
        sv = _screen_vertices(screen)
        screen_vbo.write(sv.astype("f4").tobytes())
        screen_prog["u_mvp"].write(vp.T.astype("f4").tobytes())
        screen_prog["u_color"].value = (0.1, 0.45, 1.0, 0.72)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.CULL_FACE)
        screen_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
