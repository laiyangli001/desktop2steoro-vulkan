from __future__ import annotations

import math

import moderngl
import numpy as np

from ..material_contract import GLTF_MATERIAL_TEXTURE_BINDINGS
from .contract import GltfMaterial, TextureBinding, TextureTransform


_DEFAULT_GLTF_SAMPLER = (9729, 9987, 10497, 10497)  # mag, min, wrapS, wrapT
_VALID_GLTF_MAG_FILTERS = {9728, 9729}
_VALID_GLTF_MIN_FILTERS = {9728, 9729, 9984, 9985, 9986, 9987}
_VALID_GLTF_WRAPS = {33071, 33648, 10497}


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return v if math.isfinite(v) else default


def _clamp_float(value, lo=0.0, hi=1.0, default=0.0):
    v = _safe_float(value, default)
    return max(lo, min(hi, v))


def _clamp_vec(values, size, default=1.0, lo=0.0, hi=1.0):
    out = [default] * size
    if isinstance(values, (list, tuple, np.ndarray)):
        for i in range(min(size, len(values))):
            out[i] = _clamp_float(values[i], lo, hi, default)
    return np.array(out, dtype=np.float32)


def _safe_nonnegative_float(value, default=1.0):
    return max(0.0, _safe_float(value, default))


def _safe_texcoord(value, default=0):
    idx = _safe_int(value, default)
    return idx if idx >= 0 else default


def _texture_index(tex_info):
    if not isinstance(tex_info, dict):
        return None
    idx = tex_info.get('index')
    return idx if isinstance(idx, int) and idx >= 0 else None


def _texture_image_id(tex_img_map, all_textures, tex_index):
    if not isinstance(tex_index, int):
        return -1
    image_id = tex_img_map.get(tex_index, -1)
    if isinstance(image_id, int) and 0 <= image_id < len(all_textures) and all_textures[image_id] is not None:
        return image_id
    return -1


def _texture_sampler(tex_sampler_map, tex_index):
    if not isinstance(tex_index, int):
        return _DEFAULT_GLTF_SAMPLER
    return tex_sampler_map.get(tex_index, _DEFAULT_GLTF_SAMPLER)


def _texture_transform(tex_info):
    if not isinstance(tex_info, dict):
        return None
    extensions = tex_info.get('extensions', {})
    if not isinstance(extensions, dict):
        return None
    transform = extensions.get('KHR_texture_transform')
    return transform if isinstance(transform, dict) else None


def _append_spec_gloss_mr_texture(all_textures, tex_img_map, spec_gloss_index, glossiness_factor, cache):
    """Convert specularGlossiness alpha into a glTF metallicRoughness texture."""
    src_id = _texture_image_id(tex_img_map, all_textures, spec_gloss_index)
    if src_id < 0:
        return -1
    glossiness = _clamp_float(glossiness_factor, 0.0, 1.0, 1.0)
    cache_key = (spec_gloss_index, glossiness)
    if cache_key in cache:
        return cache[cache_key]
    src = all_textures[src_id]
    alpha = src[:, :, 3].astype(np.float32) / 255.0
    roughness = np.clip(1.0 - alpha * glossiness, 0.0, 1.0)
    mr = np.empty_like(src)
    mr[:, :, 0] = 255
    mr[:, :, 1] = np.rint(roughness * 255.0).astype(np.uint8)
    mr[:, :, 2] = 0
    mr[:, :, 3] = 255
    mr_id = len(all_textures)
    all_textures.append(mr)
    cache[cache_key] = mr_id
    return mr_id


def normalize_gltf_sampler(sampler):
    if not isinstance(sampler, dict):
        return _DEFAULT_GLTF_SAMPLER
    mag_filter = _safe_int(sampler.get('magFilter'), _DEFAULT_GLTF_SAMPLER[0])
    min_filter = _safe_int(sampler.get('minFilter'), _DEFAULT_GLTF_SAMPLER[1])
    wrap_s = _safe_int(sampler.get('wrapS'), _DEFAULT_GLTF_SAMPLER[2])
    wrap_t = _safe_int(sampler.get('wrapT'), _DEFAULT_GLTF_SAMPLER[3])
    if mag_filter not in _VALID_GLTF_MAG_FILTERS:
        mag_filter = _DEFAULT_GLTF_SAMPLER[0]
    if min_filter not in _VALID_GLTF_MIN_FILTERS:
        min_filter = _DEFAULT_GLTF_SAMPLER[1]
    if wrap_s not in _VALID_GLTF_WRAPS:
        wrap_s = _DEFAULT_GLTF_SAMPLER[2]
    if wrap_t not in _VALID_GLTF_WRAPS:
        wrap_t = _DEFAULT_GLTF_SAMPLER[3]
    return (
        mag_filter,
        min_filter,
        wrap_s,
        wrap_t,
    )


def gltf_texture_cache_key(prefix, image_id, sampler):
    mag_filter, min_filter, wrap_s, wrap_t = normalize_gltf_sampler(sampler)
    return f"{prefix}:{int(image_id)}:{mag_filter}:{min_filter}:{wrap_s}:{wrap_t}"


def apply_gltf_sampler_to_texture(texture, sampler):
    mag_filter, min_filter, wrap_s, wrap_t = normalize_gltf_sampler(sampler)
    mag_map = {
        9728: moderngl.NEAREST,
        9729: moderngl.LINEAR,
    }
    min_map = {
        9728: moderngl.NEAREST,
        9729: moderngl.LINEAR,
        9984: moderngl.NEAREST_MIPMAP_NEAREST,
        9985: moderngl.LINEAR_MIPMAP_NEAREST,
        9986: moderngl.NEAREST_MIPMAP_LINEAR,
        9987: moderngl.LINEAR_MIPMAP_LINEAR,
    }
    texture.filter = (
        min_map.get(min_filter, moderngl.LINEAR_MIPMAP_LINEAR),
        mag_map.get(mag_filter, moderngl.LINEAR),
    )
    # ModernGL exposes repeat/clamp booleans; mirrored repeat is approximated as repeat.
    texture.repeat_x = wrap_s != 33071
    texture.repeat_y = wrap_t != 33071


def _is_foliage_material_name(material_name):
    material_name_l = str(material_name or '').lower()
    return (
        'plant' in material_name_l
        or 'leaf' in material_name_l
        or 'leaves' in material_name_l
        or 'foliage' in material_name_l
        or 'grass' in material_name_l
        or 'bush' in material_name_l
        or 'tree' in material_name_l
    )


def _default_gltf_material_fields():
    return {
        'tex_id': -1,
        'base_color': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'base_sampler': _DEFAULT_GLTF_SAMPLER,
        'base_texcoord': 0,
        'base_alpha': 1.0,
        'roughness_factor': 1.0,
        'metallic_factor': 1.0,
        'emissive_factor': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'normal_tex_id': -1,
        'normal_sampler': _DEFAULT_GLTF_SAMPLER,
        'normal_texcoord': 0,
        'normal_scale': 1.0,
        'occlusion_tex_id': -1,
        'occlusion_sampler': _DEFAULT_GLTF_SAMPLER,
        'occlusion_texcoord': 0,
        'occlusion_strength': 1.0,
        'unlit': False,
        'alpha_mode': 'OPAQUE',
        'alpha_cutoff': 0.5,
        'mr_tex_id': -1,
        'mr_sampler': _DEFAULT_GLTF_SAMPLER,
        'mr_texcoord': 0,
        'emissive_tex_id': -1,
        'emissive_sampler': _DEFAULT_GLTF_SAMPLER,
        'emissive_texcoord': 0,
        'double_sided': False,
        'tex_offset': np.array([0.0, 0.0], dtype=np.float32),
        'tex_scale': np.array([1.0, 1.0], dtype=np.float32),
        'tex_rotation': 0.0,
        'foliage_mode': False,
    }


def _material_contract_from_fields(fields):
    texture_slots = {}
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS:
        image_id = int(fields.get(binding.source_tex_field, -1))
        if image_id < 0:
            continue
        transform = TextureTransform()
        if binding.role == 'base':
            transform = TextureTransform(
                offset=tuple(float(v) for v in np.asarray(fields.get('tex_offset', (0.0, 0.0)), dtype=np.float32)[:2]),
                scale=tuple(float(v) for v in np.asarray(fields.get('tex_scale', (1.0, 1.0)), dtype=np.float32)[:2]),
                rotation=float(fields.get('tex_rotation', 0.0) or 0.0),
            )
        texture_slots[binding.role] = TextureBinding(
            image_id=image_id,
            sampler=tuple(int(v) for v in fields.get(binding.source_sampler_field, _DEFAULT_GLTF_SAMPLER)),
            texcoord=_safe_texcoord(fields.get(f'{binding.role}_texcoord'), 0),
            transform=transform,
            color_space=binding.color_space,
        )
    return GltfMaterial(
        base_color=tuple(float(v) for v in np.asarray(fields.get('base_color', (1.0, 1.0, 1.0)), dtype=np.float32)[:3]),
        base_alpha=float(fields.get('base_alpha', 1.0)),
        alpha_mode=fields.get('alpha_mode', 'OPAQUE'),
        alpha_cutoff=float(fields.get('alpha_cutoff', 0.5)),
        double_sided=bool(fields.get('double_sided', False)),
        unlit=bool(fields.get('unlit', False)),
        texture_slots=texture_slots,
        roughness=float(fields.get('roughness_factor', 1.0)),
        metallic=float(fields.get('metallic_factor', 1.0)),
        normal_scale=float(fields.get('normal_scale', 1.0)),
        occlusion_strength=float(fields.get('occlusion_strength', 1.0)),
        emissive_factor=tuple(float(v) for v in np.asarray(fields.get('emissive_factor', (0.0, 0.0, 0.0)), dtype=np.float32)[:3]),
    )


def _set_texture_transform(fields, tex_info, texcoord_key):
    tx_ext = _texture_transform(tex_info)
    if not tx_ext:
        return
    if 'texCoord' in tx_ext:
        fields[texcoord_key] = _safe_texcoord(tx_ext.get('texCoord'), fields[texcoord_key])
    if texcoord_key != 'base_texcoord':
        return
    if isinstance(tx_ext.get('offset'), (list, tuple)) and len(tx_ext['offset']) >= 2:
        fields['tex_offset'] = np.array([
            _safe_float(tx_ext['offset'][0], 0.0),
            _safe_float(tx_ext['offset'][1], 0.0),
        ], dtype=np.float32)
    if isinstance(tx_ext.get('scale'), (list, tuple)) and len(tx_ext['scale']) >= 2:
        fields['tex_scale'] = np.array([
            _safe_float(tx_ext['scale'][0], 1.0),
            _safe_float(tx_ext['scale'][1], 1.0),
        ], dtype=np.float32)
    if 'rotation' in tx_ext:
        fields['tex_rotation'] = _safe_float(tx_ext.get('rotation'), 0.0)


def parse_gltf_material(
    gltf,
    material_index,
    *,
    tex_img_map,
    tex_sampler_map,
    all_textures,
    uv_min=None,
    uv_max=None,
    spec_gloss_mr_cache=None,
    log_writer=None,
):
    """Parse a glTF material into renderer-facing primitive fields."""
    fields = _default_gltf_material_fields()
    materials = gltf.get('materials', [])
    if not isinstance(material_index, int) or material_index < 0 or material_index >= len(materials):
        fields['material_contract'] = _material_contract_from_fields(fields)
        return fields

    mat = materials[material_index]
    if not isinstance(mat, dict):
        mat = {}
    mat_name = mat.get('name', f'material_{material_index}')
    pbr = mat.get('pbrMetallicRoughness', {})
    pbr = pbr if isinstance(pbr, dict) else {}
    ext = mat.get('extensions', {})
    ext = ext if isinstance(ext, dict) else {}
    sg = ext.get('KHR_materials_pbrSpecularGlossiness')
    sg = sg if isinstance(sg, dict) else None

    bt = pbr.get('baseColorTexture')
    tex_index = _texture_index(bt)
    if tex_index is None and sg:
        tex_index = _texture_index(sg.get('diffuseTexture'))
    if tex_index is not None:
        tid = _texture_image_id(tex_img_map, all_textures, tex_index)
        if tid >= 0:
            fields['tex_id'] = tid
            fields['base_sampler'] = _texture_sampler(tex_sampler_map, tex_index)
            if isinstance(bt, dict):
                fields['base_texcoord'] = _safe_texcoord(bt.get('texCoord'), 0)
    if isinstance(bt, dict):
        _set_texture_transform(fields, bt, 'base_texcoord')

    bcf = pbr.get('baseColorFactor')
    if bcf is not None:
        base_rgba = _clamp_vec(bcf, 4, default=1.0, lo=0.0, hi=1.0)
        fields['base_color'] = base_rgba[:3]
        fields['base_alpha'] = float(base_rgba[3])
    rf = pbr.get('roughnessFactor')
    if rf is not None:
        fields['roughness_factor'] = _clamp_float(rf, 0.0, 1.0, fields['roughness_factor'])
    mf = pbr.get('metallicFactor')
    if mf is not None:
        fields['metallic_factor'] = _clamp_float(mf, 0.0, 1.0, fields['metallic_factor'])

    mrt = pbr.get('metallicRoughnessTexture')
    mrt_index = _texture_index(mrt)
    if mrt_index is not None:
        mr_tid = _texture_image_id(tex_img_map, all_textures, mrt_index)
        if mr_tid >= 0:
            fields['mr_tex_id'] = mr_tid
            fields['mr_sampler'] = _texture_sampler(tex_sampler_map, mrt_index)
            fields['mr_texcoord'] = _safe_texcoord(mrt.get('texCoord'), 0)
            _set_texture_transform(fields, mrt, 'mr_texcoord')

    if sg and 'diffuseFactor' in sg and bcf is None:
        diffuse_rgba = _clamp_vec(sg['diffuseFactor'], 4, default=1.0, lo=0.0, hi=1.0)
        fields['base_color'] = diffuse_rgba[:3]
        fields['base_alpha'] = float(diffuse_rgba[3])
    if sg and mf is None:
        fields['metallic_factor'] = 0.0
    glossiness_factor = _clamp_float(sg.get('glossinessFactor', 1.0), 0.0, 1.0, 1.0) if sg else 1.0
    if sg and rf is None:
        fields['roughness_factor'] = 1.0 - glossiness_factor
    if sg:
        sgt = sg.get('specularGlossinessTexture')
        sgt_index = _texture_index(sgt)
        if sgt_index is not None:
            converted_mr_id = _append_spec_gloss_mr_texture(
                all_textures,
                tex_img_map,
                sgt_index,
                glossiness_factor,
                spec_gloss_mr_cache if spec_gloss_mr_cache is not None else {},
            )
            if converted_mr_id >= 0:
                fields['mr_tex_id'] = converted_mr_id
                fields['mr_sampler'] = _texture_sampler(tex_sampler_map, sgt_index)
                fields['mr_texcoord'] = _safe_texcoord(sgt.get('texCoord'), 0)
                fields['roughness_factor'] = 1.0
                _set_texture_transform(fields, sgt, 'mr_texcoord')

    material_name_l = str(mat_name or '').lower()
    if 'chair' in material_name_l or 'seat' in material_name_l or 'cushion' in material_name_l:
        fields['metallic_factor'] = 0.0

    nt = mat.get('normalTexture')
    nt_index = _texture_index(nt)
    if nt_index is not None:
        n_tid = _texture_image_id(tex_img_map, all_textures, nt_index)
        if n_tid >= 0:
            fields['normal_tex_id'] = n_tid
            fields['normal_sampler'] = _texture_sampler(tex_sampler_map, nt_index)
            fields['normal_texcoord'] = _safe_texcoord(nt.get('texCoord'), 0)
            _set_texture_transform(fields, nt, 'normal_texcoord')
        ns = nt.get('scale')
        if ns is not None:
            fields['normal_scale'] = _safe_nonnegative_float(ns, fields['normal_scale'])

    ot = mat.get('occlusionTexture')
    ot_index = _texture_index(ot)
    if ot_index is not None:
        o_tid = _texture_image_id(tex_img_map, all_textures, ot_index)
        if o_tid >= 0:
            fields['occlusion_tex_id'] = o_tid
            fields['occlusion_sampler'] = _texture_sampler(tex_sampler_map, ot_index)
            fields['occlusion_texcoord'] = _safe_texcoord(ot.get('texCoord'), 0)
            _set_texture_transform(fields, ot, 'occlusion_texcoord')
        os_ = ot.get('strength')
        if os_ is not None:
            fields['occlusion_strength'] = _clamp_float(os_, 0.0, 1.0, fields['occlusion_strength'])

    fields['unlit'] = 'KHR_materials_unlit' in ext
    alpha_mode = mat.get('alphaMode', 'OPAQUE')
    fields['alpha_mode'] = alpha_mode if alpha_mode in ('OPAQUE', 'MASK', 'BLEND') else 'OPAQUE'
    fields['alpha_cutoff'] = _clamp_float(mat.get('alphaCutoff'), 0.0, 1.0, 0.5)

    fields['double_sided'] = bool(mat.get('doubleSided', False))
    fields['foliage_mode'] = _is_foliage_material_name(mat_name)
    if not fields['double_sided'] and fields['alpha_mode'] == 'OPAQUE' and fields['foliage_mode']:
        fields['double_sided'] = True
    if fields['tex_id'] >= 0 and fields['foliage_mode'] and uv_min is not None and uv_max is not None:
        mag_filter, min_filter, wrap_s, wrap_t = fields['base_sampler']
        if uv_min[0] < -0.05 or uv_max[0] > 1.05:
            wrap_s = 10497
        if uv_min[1] < -0.05 or uv_max[1] > 1.05:
            wrap_t = 10497
        fields['base_sampler'] = (mag_filter, min_filter, wrap_s, wrap_t)

    ef = mat.get('emissiveFactor')
    if ef is not None:
        fields['emissive_factor'] = _clamp_vec(ef, 3, default=0.0, lo=0.0, hi=1.0)
        es_ext = ext.get('KHR_materials_emissive_strength')
        if es_ext and 'emissiveStrength' in es_ext:
            fields['emissive_factor'] = fields['emissive_factor'] * _safe_nonnegative_float(es_ext['emissiveStrength'], 1.0)
    et = mat.get('emissiveTexture')
    et_index = _texture_index(et)
    if et_index is not None:
        e_tid = _texture_image_id(tex_img_map, all_textures, et_index)
        if e_tid >= 0:
            fields['emissive_tex_id'] = e_tid
            fields['emissive_sampler'] = _texture_sampler(tex_sampler_map, et_index)
            fields['emissive_texcoord'] = _safe_texcoord(et.get('texCoord'), 0)
            _set_texture_transform(fields, et, 'emissive_texcoord')

    if log_writer is not None and material_index < 300:
        emissive_info = f" emissive={fields['emissive_factor'].tolist()}" if fields['emissive_factor'].any() else ''
        log_writer.write(f"[MAT] {material_index}: {mat_name}  "
                         f"bcf={bcf}  rough={rf}  "
                         f"tex_index={tex_index}  tex_id={fields['tex_id']}"
                         f"{emissive_info}  "
                         f"ext={list(ext.keys())}\n")
    fields['material_contract'] = _material_contract_from_fields(fields)
    return fields


__all__ = [
    "apply_gltf_sampler_to_texture",
    "gltf_texture_cache_key",
    "normalize_gltf_sampler",
    "parse_gltf_material",
]
