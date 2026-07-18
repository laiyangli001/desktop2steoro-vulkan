# Desktop2Stereo OpenXR viewer: XR pose, matrix, and projection helpers.

import math

import numpy as np

try:
    import xr
except ImportError:
    xr = None

__all__ = [
    '_xr_quat_to_mat4',
    '_pose_to_view_mat4',
    'xr_pose_to_model_mat4',
    'euler_to_mat4',
    '_mat3_to_quat_xyzw',
    'mat4_to_xr_posef',
    '_fov_to_proj_mat4',
    '_fov_to_proj_mat4_d3d',
]


def _xr_quat_to_mat4(q):
    """XrQuaternionf ->standard 4x4 rotation matrix (numpy, math row/col convention).

    Produces the matrix that left-multiplies a column vector: v' = R @ v.
    Callers must transpose before writing to OpenGL (which reads column-major).
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),  0],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),  0],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),  0],
        [  0,             0,             0,              1],
    ], dtype=np.float32)


def _pose_to_view_mat4(pose):
    """XrPosef ->standard 4x4 view matrix (numpy, math row/col convention).

    The view matrix is the inverse of the head-pose model matrix:
    V = [ R^T | -R^T @ pos ]
        [  0  |      1     ]
    Caller must transpose before writing to OpenGL.
    """
    R  = _xr_quat_to_mat4(pose.orientation)[:3, :3]
    Rt = R.T                                              # inverse rotation
    t  = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    V  = np.eye(4, dtype=np.float32)
    V[:3, :3] = Rt
    V[:3, 3]  = -Rt @ t                                  # translation in last column
    return V


def xr_pose_to_model_mat4(pose):
    """XrPosef -> standard 4x4 model matrix."""
    M = _xr_quat_to_mat4(pose.orientation)
    M[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    return M


def euler_to_mat4(yaw, pitch, roll):
    """Yaw/pitch/roll radians -> 4x4 rotation matrix."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [-sy, 0.0, cy, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, cp, -sp, 0.0],
                   [0.0, sp, cp, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0, 0.0],
                   [sr, cr, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return ry @ rx @ rz


def _mat3_to_quat_xyzw(m33):
    """3x3 rotation matrix -> normalized quaternion (x, y, z, w)."""
    t = m33[0, 0] + m33[1, 1] + m33[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m33[2, 1] - m33[1, 2]) / s
        y = (m33[0, 2] - m33[2, 0]) / s
        z = (m33[1, 0] - m33[0, 1]) / s
    elif m33[0, 0] > m33[1, 1] and m33[0, 0] > m33[2, 2]:
        s = np.sqrt(1.0 + m33[0, 0] - m33[1, 1] - m33[2, 2]) * 2.0
        w = (m33[2, 1] - m33[1, 2]) / s
        x = 0.25 * s
        y = (m33[0, 1] + m33[1, 0]) / s
        z = (m33[0, 2] + m33[2, 0]) / s
    elif m33[1, 1] > m33[2, 2]:
        s = np.sqrt(1.0 + m33[1, 1] - m33[0, 0] - m33[2, 2]) * 2.0
        w = (m33[0, 2] - m33[2, 0]) / s
        x = (m33[0, 1] + m33[1, 0]) / s
        y = 0.25 * s
        z = (m33[1, 2] + m33[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m33[2, 2] - m33[0, 0] - m33[1, 1]) * 2.0
        w = (m33[1, 0] - m33[0, 1]) / s
        x = (m33[0, 2] + m33[2, 0]) / s
        y = (m33[1, 2] + m33[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def mat4_to_xr_posef(mat4):
    """4x4 rigid transform -> XrPosef."""
    q = _mat3_to_quat_xyzw(mat4[:3, :3].astype(np.float64))
    pose = xr.Posef()
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    pose.position.x = float(mat4[0, 3])
    pose.position.y = float(mat4[1, 3])
    pose.position.z = float(mat4[2, 3])
    return pose


def _fov_to_proj_mat4(fov, near=0.05, far=100.0):
    """XrFovf ->standard 4x4 OpenGL asymmetric-frustum projection matrix
    (numpy, math row/col convention). Caller must transpose before writing to OpenGL.

    Includes a small epsilon offset to prevent division by zero when the
    headset runtime reports a degenerate FOV (e.g., left == right).
    """
    l = math.tan(fov.angle_left)  * near
    r = math.tan(fov.angle_right) * near
    t = math.tan(fov.angle_up)    * near
    b = math.tan(fov.angle_down)  * near

    # Prevent ZeroDivisionError when headset reports identical left/right or up/down angles.
    EPS = 1e-6
    if abs(r - l) < EPS:
        r += EPS
    if abs(t - b) < EPS:
        t += EPS

    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] =  2 * near / (r - l)
    p[0, 2] =  (r + l)  / (r - l)      # col 2 of row 0
    p[1, 1] =  2 * near / (t - b)
    p[1, 2] =  (t + b)  / (t - b)      # col 2 of row 1
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)  # translation in last column
    p[3, 2] = -1.0                      # w = -z (perspective divide)
    return p


def _fov_to_proj_mat4_d3d(fov, near=0.05, far=100.0):
    """XrFovf ->D3D-style asymmetric-frustum projection matrix.

    D3D clip-space depth is 0..1. The view convention still looks down -Z,
    matching _pose_to_view_mat4 and the existing OpenGL world transforms.
    """
    l = math.tan(fov.angle_left)  * near
    r = math.tan(fov.angle_right) * near
    t = math.tan(fov.angle_up)    * near
    b = math.tan(fov.angle_down)  * near

    EPS = 1e-6
    if abs(r - l) < EPS:
        r += EPS
    if abs(t - b) < EPS:
        t += EPS

    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] =  2 * near / (r - l)
    p[0, 2] =  (r + l)  / (r - l)
    p[1, 1] =  2 * near / (t - b)
    p[1, 2] =  (t + b)  / (t - b)
    p[2, 2] = -far / (far - near)
    p[2, 3] = -(far * near) / (far - near)
    p[3, 2] = -1.0
    return p
