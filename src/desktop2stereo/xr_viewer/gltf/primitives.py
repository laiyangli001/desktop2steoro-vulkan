"""glTF primitive, texture, and light loading helpers."""

import io as _io
import os
import urllib.parse

import moderngl
import numpy as np
from PIL import Image

from .accessors import _coerce_vec_array, _get_accessor
from .contract import attach_primitive_contract
from .document import _buffer_data, _decode_data_uri, _load_gltf_document
from .materials import _safe_int, normalize_gltf_sampler, parse_gltf_material
from .scene import (
    _apply_normal_transform,
    _apply_transform,
    _build_node_matrices,
    _iter_scene_mesh_nodes,
    _node_local_matrix,
    _orthogonalize_tangent,
)
from .validation import raise_unsupported_required_extensions as _raise_unsupported_required_extensions
from ..material_contract import GLTF_MATERIAL_TEXTURE_BINDINGS

def gltf_primitive_mode_to_moderngl(mode):
    return {
        0: moderngl.POINTS,
        1: moderngl.LINES,
        2: moderngl.LINE_LOOP,
        3: moderngl.LINE_STRIP,
        4: moderngl.TRIANGLES,
        5: moderngl.TRIANGLE_STRIP,
        6: moderngl.TRIANGLE_FAN,
    }.get(int(mode), moderngl.TRIANGLES)


def load_glb_model(path):
    """Load a glTF/GLB model, apply node transformations.
    Returns:
        primitives: list of dict with keys:
            vertices (N, 10 float32: position3, normal3, uv0, uv1)
            tangent (N, 4 float32: tangent xyz, bitangent sign)
            indices (M, uint32)
            material_contract (GltfMaterial)
            gltf_primitive (GltfPrimitive)
            render_pass (opaque, mask, transparent, or sky)
        textures: list of numpy RGBA uint8 arrays
    """
    _mat_log = open(os.devnull, 'w', encoding='utf-8')
    _mat_log.write(f"=== Material debug for: {path} ===\n")
    gltf, bin_data = _load_gltf_document(path)
    _raise_unsupported_required_extensions(gltf, path)
    base_dir = os.path.dirname(os.path.abspath(path))

    # World matrices for all nodes
    world_mats = _build_node_matrices(gltf)
    nodes = gltf.get('nodes', [])
    local_mats = [_node_local_matrix(node) for node in nodes]
    parent = [-1] * len(nodes)
    name_to_node = {}
    for ni, node in enumerate(nodes):
        name = node.get('name')
        if name:
            name_to_node[str(name)] = ni
        for child in node.get('children', []):
            if isinstance(child, int) and 0 <= child < len(nodes):
                parent[child] = ni

    def _semantic_from_value_name(value_name, node_name=''):
        value_l = value_name.lower()
        node_l = node_name.lower()
        if 'thumbstick_xaxis' in value_l:
            return 'joystick_x'
        if 'thumbstick_yaxis' in value_l:
            return 'joystick_y'
        if 'thumbstick_pressed' in value_l:
            return 'joystick'
        if 'thumbstick' in value_l and 'touched' in value_l:
            return 'joystick_touched'
        if 'touchpad_xaxis' in value_l:
            return 'touchpad_x'
        if 'touchpad_yaxis' in value_l:
            return 'touchpad_y'
        if 'touchpad_pressed' in value_l:
            return 'touchpad'
        if 'touchpad' in value_l and 'touched' in value_l:
            return 'touchpad_touched'
        if 'trigger' in value_l:
            return 'trigger'
        if 'squeeze' in value_l or 'grasp' in value_l or 'grip' in node_l:
            return 'grip'
        if 'a_button' in value_l or node_l.endswith('abutton') or node_l in ('a_button', 'a_button_mesh'):
            return 'a_button'
        if 'b_button' in value_l or node_l.endswith('bbutton') or node_l in ('b_button', 'b_button_mesh'):
            return 'b_button'
        if 'x_button' in value_l or node_l.endswith('xbutton') or node_l in ('x_button', 'x_button_mesh'):
            return 'x_button'
        if 'y_button' in value_l or node_l.endswith('ybutton') or node_l in ('y_button', 'y_button_mesh'):
            return 'y_button'
        if 'menu' in value_l or 'menu' in node_l:
            return 'menu_button'
        if 'home' in value_l or 'pico' in value_l or 'home' in node_l:
            return 'home_button'
        return ''

    def _anim_from_value_node(node_index, value_node_index):
        value_name = str(nodes[value_node_index].get('name') or '')
        if not value_name.endswith('_value'):
            return None
        prefix = value_name[:-len('_value')]
        min_index = name_to_node.get(prefix + '_min')
        max_index = name_to_node.get(prefix + '_max')
        if min_index is None or max_index is None:
            return None
        mesh_world = world_mats[node_index].astype(np.float32)
        value_world = world_mats[value_node_index].astype(np.float32)
        try:
            child_local = (np.linalg.inv(value_world.astype(np.float64)) @ mesh_world.astype(np.float64)).astype(np.float32)
            inv_mesh_world = np.linalg.inv(mesh_world.astype(np.float64)).astype(np.float32)
        except Exception:
            return None
        parent_index = parent[value_node_index] if value_node_index < len(parent) else -1
        value_parent_world = world_mats[parent_index] if parent_index >= 0 else np.eye(4, dtype=np.float64)
        return {
            'value_name': value_name,
            'semantic': _semantic_from_value_name(value_name, str(nodes[node_index].get('name') or '')),
            'value_world': world_mats[value_node_index].astype(np.float32),
            'min_world': world_mats[min_index].astype(np.float32),
            'max_world': world_mats[max_index].astype(np.float32),
            'value_parent_world': value_parent_world.astype(np.float32),
            'value_local': local_mats[value_node_index].astype(np.float32),
            'min_local': local_mats[min_index].astype(np.float32),
            'max_local': local_mats[max_index].astype(np.float32),
            'child_local': child_local,
            'inv_mesh_world': inv_mesh_world,
        }

    def _press_anim_for_mesh_node(node_index):
        node_name = str(nodes[node_index].get('name') or '') if 0 <= node_index < len(nodes) else ''
        node_l = node_name.lower()
        candidates = []
        parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
        while parent_index >= 0:
            value_name = str(nodes[parent_index].get('name') or '')
            if value_name.endswith('_value'):
                prefix = value_name[:-len('_value')]
                min_index = name_to_node.get(prefix + '_min')
                max_index = name_to_node.get(prefix + '_max')
                if min_index is not None and max_index is not None:
                    candidates.append((parent_index, value_name, min_index, max_index))
            parent_index = parent[parent_index] if parent_index < len(parent) else -1
        if not candidates:
            return None
        is_stick = (
            'joystick' in node_l
            or 'thumbstick' in node_l
            or 'touchpad' in node_l
            or any(('thumbstick' in candidate[1].lower() or 'touchpad' in candidate[1].lower()) for candidate in candidates)
        )
        if is_stick:
            for candidate in candidates:
                if candidate[1].endswith('thumbstick_pressed_value') or candidate[1].endswith('touchpad_pressed_value'):
                    return _anim_from_value_node(node_index, candidate[0])
            return None
        return _anim_from_value_node(node_index, candidates[0][0])

    def _visible_key_for_mesh_node(node_index):
        parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
        while parent_index >= 0:
            value_name = str(nodes[parent_index].get('name') or '')
            if value_name.endswith('_value') and 'touched' in value_name.lower():
                return _semantic_from_value_name(value_name, str(nodes[node_index].get('name') or ''))
            parent_index = parent[parent_index] if parent_index < len(parent) else -1
        return ''

    def _axis_anim_for_mesh_node(node_index):
        node_name = str(nodes[node_index].get('name') or '') if 0 <= node_index < len(nodes) else ''
        node_l = node_name.lower()
        if 'joystick' not in node_l and 'thumbstick' not in node_l and 'touchpad' not in node_l:
            parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
            has_stick_axis = False
            while parent_index >= 0:
                value_name = str(nodes[parent_index].get('name') or '').lower()
                if 'thumbstick_' in value_name or 'touchpad_' in value_name:
                    has_stick_axis = True
                    break
                parent_index = parent[parent_index] if parent_index < len(parent) else -1
            if not has_stick_axis:
                return None
        parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
        result = {}
        while parent_index >= 0:
            value_name = str(nodes[parent_index].get('name') or '')
            if value_name.endswith('thumbstick_xaxis_pressed_value') or value_name.endswith('touchpad_xaxis_pressed_value'):
                result['x'] = _anim_from_value_node(node_index, parent_index)
            elif value_name.endswith('thumbstick_yaxis_pressed_value') or value_name.endswith('touchpad_yaxis_pressed_value'):
                result['y'] = _anim_from_value_node(node_index, parent_index)
            parent_index = parent[parent_index] if parent_index < len(parent) else -1
        return result or None

    # Map mesh index to all node instances that reference it.  glTF allows
    # table legs, curtains, string lights, etc. to reuse one mesh from many
    # nodes; the old loader kept only the first node and dropped/misplaced the
    # rest.
    mesh_world_mat = {}
    mesh_world_mats = {}
    mesh_node_meta = {}
    for mi, world_mat_for_node, node_index, node_name, mesh_name in _iter_scene_mesh_nodes(gltf, world_mats):
        press_anim = _press_anim_for_mesh_node(node_index)
        axis_anim = _axis_anim_for_mesh_node(node_index)
        visible_key = _visible_key_for_mesh_node(node_index)
        mesh_world_mats.setdefault(mi, []).append((world_mat_for_node, node_index, node_name, mesh_name, press_anim, axis_anim, visible_key))
        if mi not in mesh_world_mat:
            mesh_world_mat[mi] = world_mat_for_node
            mesh_node_meta[mi] = (node_index, node_name, mesh_name, press_anim, axis_anim, visible_key)

    # Extract textures
    all_textures = []
    if 'images' in gltf:
        for img in gltf['images']:
            tex_data = None
            if isinstance(img, dict) and 'bufferView' in img:
                buffer_views = gltf.get('bufferViews', [])
                bv_idx = img.get('bufferView')
                if isinstance(bv_idx, int) and 0 <= bv_idx < len(buffer_views):
                    bv = buffer_views[bv_idx]
                    off = _safe_int(bv.get('byteOffset'), 0)
                    byte_len = _safe_int(bv.get('byteLength'), 0)
                    image_buffer = _buffer_data(bin_data, bv.get('buffer', 0))
                    if image_buffer is not None and off >= 0 and byte_len > 0 and off + byte_len <= len(image_buffer):
                        tex_data = image_buffer[off:off + byte_len]
            elif isinstance(img, dict) and 'uri' in img and img['uri'].startswith('data:'):
                tex_data = _decode_data_uri(img['uri'])
            elif isinstance(img, dict) and 'uri' in img:
                uri = img['uri']
                parsed = urllib.parse.urlparse(uri)
                if parsed.scheme in ('', 'file'):
                    rel_path = urllib.parse.unquote(parsed.path if parsed.scheme == 'file' else uri)
                    rel_path = rel_path.replace('/', os.sep)
                    tex_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
                    if os.path.exists(tex_path):
                        with open(tex_path, 'rb') as tf:
                            tex_data = tf.read()
            if tex_data:
                pil_img = Image.open(_io.BytesIO(tex_data))
                pil_img = pil_img.convert('RGBA')
                all_textures.append(np.array(pil_img, dtype=np.uint8))
            else:
                all_textures.append(None)

    # Map texture index to image index
    tex_img_map = {}
    tex_sampler_map = {}
    if 'textures' in gltf:
        for ti, tex in enumerate(gltf['textures']):
            tex = tex if isinstance(tex, dict) else {}
            si = tex.get('source', 0)
            tex_img_map[ti] = si if isinstance(si, int) and 0 <= si < len(all_textures) else -1
            sampler_idx = tex.get('sampler')
            sampler = None
            if isinstance(sampler_idx, int) and 0 <= sampler_idx < len(gltf.get('samplers', [])):
                sampler = gltf['samplers'][sampler_idx]
            tex_sampler_map[ti] = normalize_gltf_sampler(sampler)
    spec_gloss_mr_cache = {}

    primitives = []
    for mi, mesh in enumerate(gltf.get('meshes', [])):
        if mi not in mesh_world_mats:
            continue
        world_mat = mesh_world_mat.get(mi, np.eye(4, dtype=np.float64))
        node_index, node_name, mesh_name, press_anim, axis_anim, visible_key = mesh_node_meta.get(mi, (-1, '', str(mesh.get('name') or ''), None, None, ''))
        for prim in mesh.get('primitives', []):
            attrs = prim.get('attributes', {})
            if 'POSITION' not in attrs:
                continue
            try:
                pos = _get_accessor(gltf, bin_data, attrs['POSITION'])
            except Exception as exc:
                _mat_log.write(f"[PRIM] skip mesh={mi}: invalid POSITION ({exc})\n")
                continue
            if pos.ndim != 2 or pos.shape[1] < 3 or pos.shape[0] == 0:
                _mat_log.write(f"[PRIM] skip mesh={mi}: POSITION must be non-empty VEC3\n")
                continue
            pos = pos[:, :3].astype(np.float32, copy=False)

            # Extract normals if present, else zeros
            if 'NORMAL' in attrs:
                try:
                    norm = _get_accessor(gltf, bin_data, attrs['NORMAL'])
                except Exception:
                    norm = np.zeros((pos.shape[0], 3), dtype=np.float32)
            else:
                norm = np.zeros((pos.shape[0], 3), dtype=np.float32)
            norm = _coerce_vec_array(norm, pos.shape[0], 3, 0.0)

            # Extract tangent (vec4: xyz + bitangent_sign), or zeros if absent
            if 'TANGENT' in attrs:
                try:
                    tangent = _get_accessor(gltf, bin_data, attrs['TANGENT'])
                except Exception:
                    tangent = np.zeros((pos.shape[0], 4), dtype=np.float32)
                    tangent[:, 3] = 1.0
                tangent = _coerce_vec_array(tangent, pos.shape[0], 4, 0.0)
                if tangent.shape[0] > 0:
                    tangent[:, 3] = np.where(np.abs(tangent[:, 3]) > 1e-8, tangent[:, 3], 1.0)
            else:
                tangent = np.zeros((pos.shape[0], 4), dtype=np.float32)
                tangent[:, 3] = 1.0  # bitangent sign defaults to 1

            # Apply node world matrix: position with full 4x4, normals with inverse-transpose
            if not np.allclose(world_mat, np.eye(4)):
                pos = _apply_transform(pos, world_mat)
                rot3 = world_mat[:3, :3].astype(np.float64)
                normal_mat = np.linalg.inv(rot3.T)  # inverse-transpose handles non-uniform scaling
                norm = (normal_mat @ norm.T).T.astype(np.float32)
                norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-8)
                # Transform tangent xyz with rotation, keep w (bitangent sign)
                if tangent is not None:
                    t_xyz = (rot3[:3, :3].astype(np.float64) @ tangent[:, :3].T).T.astype(np.float32)
                    t_xyz = _orthogonalize_tangent(t_xyz, norm)
                    tangent = np.hstack([t_xyz, tangent[:, 3:4]]).astype(np.float32)

            # Extract UV coordinates. Keep UV1 for glTF textureInfo.texCoord=1 lightmaps.
            if 'TEXCOORD_0' in attrs:
                try:
                    uv = _get_accessor(gltf, bin_data, attrs['TEXCOORD_0'])
                except Exception:
                    uv = np.zeros((pos.shape[0], 2), dtype=np.float32)
            else:
                uv = np.zeros((pos.shape[0], 2), dtype=np.float32)
            uv = _coerce_vec_array(uv, pos.shape[0], 2, 0.0)

            has_uv1 = 'TEXCOORD_1' in attrs
            if has_uv1:
                try:
                    uv1 = _get_accessor(gltf, bin_data, attrs['TEXCOORD_1'])
                except Exception:
                    uv1 = uv.copy()
            else:
                uv1 = uv.copy()
            uv1 = _coerce_vec_array(uv1, pos.shape[0], 2, 0.0)

            uv_min = uv.min(axis=0) if uv.size else np.array([0.0, 0.0], dtype=np.float32)
            uv_max = uv.max(axis=0) if uv.size else np.array([0.0, 0.0], dtype=np.float32)

            # Combine: position (3), normal (3), uv0 (2), uv1 (2) -> 10 floats
            vertices = np.hstack([pos, norm, uv, uv1]).astype(np.float32)

            # Indices
            if 'indices' in prim:
                try:
                    indices = _get_accessor(gltf, bin_data, prim['indices']).reshape(-1).astype(np.uint32, copy=False)
                except Exception:
                    indices = np.arange(pos.shape[0], dtype=np.uint32)
            else:
                indices = np.arange(pos.shape[0], dtype=np.uint32)
            if indices.size == 0 or int(indices.max()) >= pos.shape[0]:
                indices = np.arange(pos.shape[0], dtype=np.uint32)

            material_fields = parse_gltf_material(
                gltf,
                prim.get('material'),
                tex_img_map=tex_img_map,
                tex_sampler_map=tex_sampler_map,
                all_textures=all_textures,
                uv_min=uv_min,
                uv_max=uv_max,
                spec_gloss_mr_cache=spec_gloss_mr_cache,
                log_writer=_mat_log,
            )
            primitive_record = {'vertices': vertices, 'indices': indices,
                                'primitive_mode': _safe_int(prim.get('mode'), 4),
                                'has_uv1': has_uv1,
                                'tangent': tangent,
                                'node_index': node_index,
                                'node_name': node_name,
                                'mesh_name': mesh_name,
                                'press_anim': press_anim,
                                'axis_anim': axis_anim,
                                'anim_key': press_anim.get('semantic', '') if press_anim else '',
                                'visible_key': visible_key,
                                '_mesh_index': mi,
                                '_world_matrix': world_mat}
            primitive_record.update(material_fields)
            primitives.append(primitive_record)

    extra_instances = []
    for primitive in primitives:
        mi = primitive.get('_mesh_index')
        instances = mesh_world_mats.get(mi, [])
        if len(instances) <= 1:
            continue

        first_world = primitive.get('_world_matrix', np.eye(4, dtype=np.float64)).astype(np.float64)
        try:
            inv_first_world = np.linalg.inv(first_world)
        except Exception:
            continue

        local_positions = _apply_transform(primitive['vertices'][:, :3], inv_first_world)
        first_rot = first_world[:3, :3].astype(np.float64)
        local_normals = (first_rot.T @ primitive['vertices'][:, 3:6].astype(np.float64).T).T
        local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-8)

        tangent = primitive.get('tangent')
        if tangent is not None:
            local_tangent = tangent.copy()
            local_tangent[:, :3] = (first_rot.T @ tangent[:, :3].astype(np.float64).T).T.astype(np.float32)
            local_tangent[:, :3] = _orthogonalize_tangent(local_tangent[:, :3], local_normals)
        else:
            local_tangent = None

        for inst_world, node_index, node_name, mesh_name, press_anim, axis_anim, visible_key in instances[1:]:
            inst_world = inst_world.astype(np.float64)
            clone = dict(primitive)
            clone_vertices = primitive['vertices'].copy()
            clone_vertices[:, :3] = _apply_transform(local_positions, inst_world)
            clone_vertices[:, 3:6] = _apply_normal_transform(local_normals, inst_world)
            clone['vertices'] = clone_vertices
            clone['indices'] = primitive['indices'].copy()
            if local_tangent is not None:
                inst_tangent = local_tangent.copy()
                inst_tangent[:, :3] = (inst_world[:3, :3].astype(np.float64) @ local_tangent[:, :3].astype(np.float64).T).T.astype(np.float32)
                inst_tangent[:, :3] = _orthogonalize_tangent(inst_tangent[:, :3], clone_vertices[:, 3:6])
                clone['tangent'] = inst_tangent
            clone['_world_matrix'] = inst_world
            clone['node_index'] = node_index
            clone['node_name'] = node_name
            clone['mesh_name'] = mesh_name
            clone['press_anim'] = press_anim
            clone['axis_anim'] = axis_anim
            clone['anim_key'] = press_anim.get('semantic', '') if press_anim else ''
            clone['visible_key'] = visible_key
            extra_instances.append(clone)

    if extra_instances:
        primitives.extend(extra_instances)
        _mat_log.write(f"[INSTANCE] Added {len(extra_instances)} mesh node instances\n")

    # Freeze the renderer-facing mesh/material/pass contract after all node
    # instances have received their final world-space geometry.
    for primitive in primitives:
        attach_primitive_contract(primitive)

    # Extract KHR_lights_punctual
    lights = []
    try:
        gltf_lights = gltf.get('extensions', {}).get('KHR_lights_punctual', {})
        if isinstance(gltf_lights, dict):
            gltf_lights = gltf_lights.get('lights', [])
        else:
            gltf_lights = []
        for ni, node in enumerate(gltf.get('nodes', [])):
            lext = node.get('extensions', {}).get('KHR_lights_punctual')
            if lext and 'light' in lext:
                li = lext['light']
                if li < len(gltf_lights):
                    ldef = gltf_lights[li]
                    world_mat = world_mats[ni] if ni < len(world_mats) else np.eye(4, dtype=np.float64)
                    direction = -world_mat[:3, 2].astype(np.float32)
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    position = world_mat[:3, 3].astype(np.float32)
                    spot = ldef.get('spot', {}) if isinstance(ldef.get('spot', {}), dict) else {}
                    lights.append({
                        'type': ldef.get('type', 'directional'),
                        'color': np.array(ldef.get('color', [1, 1, 1])[:3], dtype=np.float32),
                        'intensity': float(ldef.get('intensity', 1.0)),
                        'direction': direction,
                        'position': position,
                        'range': float(ldef.get('range', 0.0) or 0.0),
                        'innerConeAngle': float(spot.get('innerConeAngle', 0.0) or 0.0),
                        'outerConeAngle': float(spot.get('outerConeAngle', 0.7853981633974483) or 0.7853981633974483),
                    })
                    _mat_log.write(f"[LIGHT] {ldef.get('name', '?')}: type={ldef.get('type')} color={ldef.get('color')} intensity={ldef.get('intensity')}\n")
    except Exception as e:
        _mat_log.write(f"[LIGHT] extraction failed: {e}\n")

    _mat_log.write("=== End ===\n")
    _mat_log.close()
    return primitives, all_textures, lights


__all__ = ["gltf_primitive_mode_to_moderngl", "load_glb_model"]
