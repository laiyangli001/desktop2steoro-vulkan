"""glTF scene graph transforms and scene-level entrypoints."""

import numpy as np

from .color_management import color_management_diagnostics
from .contract import GltfMaterial, GltfScene, build_render_plan
from .document import _load_gltf_document
from .validation import raise_unsupported_required_extensions as _raise_unsupported_required_extensions

def _quat_to_mat4(q):
    """Convert quaternion [x, y, z, w] to 4x4 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),   0],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),   0],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy), 0],
        [0,           0,           0,           1],
    ], dtype=np.float64)


def _node_local_matrix(node):
    matrix = node.get('matrix')
    if isinstance(matrix, list) and len(matrix) == 16:
        # glTF stores matrices in column-major order.
        return np.array(matrix, dtype=np.float64).reshape((4, 4)).T

    t = node.get('translation', [0, 0, 0])
    r = node.get('rotation', [0, 0, 0, 1])  # [x, y, z, w]
    s = node.get('scale', [1, 1, 1])

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = t
    R = _quat_to_mat4(r)
    S_mat = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float64)
    return T @ R @ S_mat


def _build_node_matrices(gltf):
    """Compute world matrix for each node (top-down). Returns list of 4x4 float64 matrices.
    Parent world matrix = parent_matrix @ local_matrix.
    Local matrix = translation * rotation * scale.
    Root nodes assume identity parent matrix.
    """
    nodes = gltf.get('nodes', [])
    n = len(nodes)
    if n == 0:
        return []

    # Build local matrices
    local_mats = [_node_local_matrix(node) for node in nodes]

    # Build child -> parent mapping
    parent = [-1] * n
    for pi, node in enumerate(nodes):
        for ci in node.get('children', []):
            if isinstance(ci, int) and 0 <= ci < n:
                parent[ci] = pi

    # Topological order (BFS from roots) to compute world matrices
    world_mats = [None] * n
    queue = [i for i in range(n) if parent[i] == -1]
    for i in queue:
        world_mats[i] = local_mats[i].copy()

    head = 0
    while head < len(queue):
        pi = queue[head]
        head += 1
        for ci in nodes[pi].get('children', []):
            if not isinstance(ci, int) or ci < 0 or ci >= n:
                continue
            if world_mats[ci] is None:
                world_mats[ci] = world_mats[pi] @ local_mats[ci]
                queue.append(ci)

    # Isolated nodes (no parent) just use local matrix
    for i in range(n):
        if world_mats[i] is None:
            world_mats[i] = local_mats[i].copy()

    return world_mats


def _iter_scene_mesh_nodes(gltf, world_mats):
    """Yield mesh node records reachable from the active scene."""
    nodes = gltf.get('nodes', [])
    scenes = gltf.get('scenes', [])
    scene_idx = gltf.get('scene', 0)
    if isinstance(scene_idx, int) and 0 <= scene_idx < len(scenes):
        roots = scenes[scene_idx].get('nodes', [])
    else:
        roots = [i for i, node in enumerate(nodes) if node.get('mesh') is not None]

    stack = list(reversed([i for i in roots if isinstance(i, int) and 0 <= i < len(nodes)]))
    visited = set()
    while stack:
        ni = stack.pop()
        if ni in visited:
            continue
        visited.add(ni)
        node = nodes[ni]
        mi = node.get('mesh')
        if isinstance(mi, int):
            meshes = gltf.get('meshes', [])
            mesh_name = str(meshes[mi].get('name') or '') if 0 <= mi < len(meshes) else ''
            yield mi, world_mats[ni], ni, str(node.get('name') or ''), mesh_name
        children = node.get('children', [])
        stack.extend(reversed([ci for ci in children if isinstance(ci, int) and 0 <= ci < len(nodes)]))


def _apply_transform(vertices_xyz, matrix_4x4):
    """Apply 4x4 transformation matrix to vertex positions."""
    n = vertices_xyz.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    v4 = np.hstack([vertices_xyz.astype(np.float64), ones])
    t = (matrix_4x4 @ v4.T).T
    return t[:, :3].astype(np.float32)


def _apply_normal_transform(vectors_xyz, matrix_4x4):
    """Apply inverse-transpose normal transform and normalize the result."""
    rot3 = matrix_4x4[:3, :3].astype(np.float64)
    normal_mat = np.linalg.inv(rot3.T)
    out = (normal_mat @ vectors_xyz.astype(np.float64).T).T
    lens = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(lens, 1e-8)
    return out.astype(np.float32)


def _orthogonalize_tangent(tangent_xyz, normal_xyz):
    """Project tangent away from normal and normalize it."""
    tangent_xyz = tangent_xyz.astype(np.float32)
    normal_xyz = normal_xyz.astype(np.float32)
    tangent_xyz = tangent_xyz - normal_xyz * np.sum(tangent_xyz * normal_xyz, axis=1, keepdims=True)
    tangent_xyz /= np.maximum(np.linalg.norm(tangent_xyz, axis=1, keepdims=True), 1e-8)
    return tangent_xyz.astype(np.float32)


def _attach_color_management_diagnostics(diagnostics):
    merged = dict(diagnostics or {})
    merged['colorManagement'] = color_management_diagnostics()
    return merged


def load_gltf_scene(path):
    gltf, _buffers = _load_gltf_document(path)
    diagnostics = _attach_color_management_diagnostics(_raise_unsupported_required_extensions(gltf, path))
    from .primitives import load_glb_model

    primitives, textures, lights = load_glb_model(path)
    return GltfScene(
        primitives=tuple(primitives),
        textures=tuple(textures),
        lights=tuple(lights),
        render_plan=build_render_plan(primitives),
        diagnostics=diagnostics,
    )


def summarize_gltf_scene(primitives, textures, lights, diagnostics=None):
    alpha_modes = {}
    render_passes = {}
    vertex_widths = set()
    scene_min = None
    scene_max = None
    for primitive in primitives:
        material = primitive.get('material_contract') if isinstance(primitive, dict) else None
        alpha_mode = str(material.alpha_mode if isinstance(material, GltfMaterial) else 'OPAQUE').upper()
        render_pass = str(primitive.get('render_pass', '') or '')
        alpha_modes[alpha_mode] = alpha_modes.get(alpha_mode, 0) + 1
        if render_pass:
            render_passes[render_pass] = render_passes.get(render_pass, 0) + 1
        vertices = primitive.get('vertices')
        if isinstance(vertices, np.ndarray) and vertices.ndim == 2:
            vertex_widths.add(int(vertices.shape[1]))
            if vertices.shape[0] and vertices.shape[1] >= 3:
                mn = vertices[:, :3].min(axis=0).astype(np.float32)
                mx = vertices[:, :3].max(axis=0).astype(np.float32)
                scene_min = mn if scene_min is None else np.minimum(scene_min, mn)
                scene_max = mx if scene_max is None else np.maximum(scene_max, mx)

    summary = {
        'primitive_count': len(primitives),
        'texture_count': len(textures),
        'light_count': len(lights),
        'alpha_modes': dict(sorted(alpha_modes.items())),
        'render_passes': dict(sorted(render_passes.items())),
        'vertex_widths': sorted(vertex_widths),
        'scene_bounds': None,
        'diagnostics': diagnostics or {},
    }
    if scene_min is not None and scene_max is not None:
        summary['scene_bounds'] = (scene_min, scene_max)
    return summary


def format_gltf_scene_summary(summary, *, label='glTF model'):
    diagnostics = summary.get('diagnostics') or {}
    alpha_modes = summary.get('alpha_modes') or {}
    render_passes = summary.get('render_passes') or {}
    unsupported_required = diagnostics.get('unsupportedRequired') or []
    unsupported_optional = diagnostics.get('unsupportedOptional') or []
    material_extensions = diagnostics.get('materialExtensions') or []
    primitive_extensions = diagnostics.get('primitiveExtensions') or []
    vertex_widths = summary.get('vertex_widths') or []
    return (
        f"{label}: primitives={summary.get('primitive_count', 0)} "
        f"textures={summary.get('texture_count', 0)} "
        f"lights={summary.get('light_count', 0)} "
        f"vertex_widths={vertex_widths} "
        f"alpha_modes={alpha_modes} "
        f"render_passes={render_passes} "
        f"unsupported_required={unsupported_required} "
        f"unsupported_optional={unsupported_optional} "
        f"material_extensions={material_extensions} "
        f"primitive_extensions={primitive_extensions}"
    )


def diagnose_gltf_model(path):
    scene = load_gltf_scene(path)
    summary = summarize_gltf_scene(scene.primitives, scene.textures, scene.lights, scene.diagnostics)
    summary['render_plan'] = scene.render_plan
    return summary


__all__ = [
    "_apply_normal_transform",
    "_apply_transform",
    "_build_node_matrices",
    "_iter_scene_mesh_nodes",
    "_node_local_matrix",
    "_orthogonalize_tangent",
    "_quat_to_mat4",
    "diagnose_gltf_model",
    "format_gltf_scene_summary",
    "load_gltf_scene",
    "summarize_gltf_scene",
]
