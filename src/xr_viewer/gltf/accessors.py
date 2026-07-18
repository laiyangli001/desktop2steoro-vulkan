"""glTF accessor decoding helpers."""

import numpy as np

from .document import _buffer_data
from .materials import _safe_int


_DTYPE_MAP = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
            5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
            'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


def _coerce_vec_array(values, rows, cols, fill=0.0):
    out = np.full((rows, cols), fill, dtype=np.float32)
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return out
    if arr.ndim != 2 or arr.shape[0] != rows:
        return out
    ncols = min(cols, arr.shape[1])
    if ncols > 0:
        out[:, :ncols] = arr[:, :ncols]
    return out


def _get_accessor(gltf, bin_data, acc_idx):
    """Extract numpy array from a glTF accessor.
    Handles both contiguous and interleaved (byteStride) vertex attributes.
    """
    accessors = gltf.get('accessors', [])
    if not isinstance(accessors, list) or not isinstance(acc_idx, int) or acc_idx < 0 or acc_idx >= len(accessors):
        raise ValueError(f"Invalid accessor index: {acc_idx}")
    acc = accessors[acc_idx]
    if not isinstance(acc, dict):
        raise ValueError(f"Invalid accessor object: {acc_idx}")
    if acc.get('type') not in _TYPE_NC:
        raise ValueError(f"Unsupported accessor type: {acc.get('type')}")
    if acc.get('componentType') not in _DTYPE_MAP:
        raise ValueError(f"Unsupported accessor componentType: {acc.get('componentType')}")
    count = _safe_int(acc.get('count'), -1)
    if count < 0:
        raise ValueError(f"Invalid accessor count: {acc.get('count')}")
    nc = _TYPE_NC[acc['type']]
    dt = np.dtype(_DTYPE_MAP[acc['componentType']]).newbyteorder('<')
    elem_size = nc * dt.itemsize

    if 'bufferView' in acc:
        buffer_views = gltf.get('bufferViews', [])
        bv_idx = acc.get('bufferView')
        if not isinstance(buffer_views, list) or not isinstance(bv_idx, int) or bv_idx < 0 or bv_idx >= len(buffer_views):
            raise ValueError(f"Invalid bufferView index: {bv_idx}")
        bv = buffer_views[bv_idx]
        if not isinstance(bv, dict):
            raise ValueError(f"Invalid bufferView object: {bv_idx}")
        byte_offset = _safe_int(bv.get('byteOffset'), 0) + _safe_int(acc.get('byteOffset'), 0)
        byte_stride = _safe_int(bv.get('byteStride'), 0)
        if byte_offset < 0 or byte_stride < 0:
            raise ValueError("Negative accessor byte offset or stride")
        if byte_stride and byte_stride < elem_size:
            raise ValueError(f"Accessor byteStride smaller than element size: {byte_stride} < {elem_size}")
        required_bytes = elem_size * count if (byte_stride == 0 or byte_stride == elem_size or count == 0) else byte_stride * (count - 1) + elem_size
        buffer_data = _buffer_data(bin_data, bv.get('buffer', 0))
        if buffer_data is None or byte_offset + required_bytes > len(buffer_data):
            raise ValueError("Accessor buffer range exceeds buffer data")
        if byte_stride == 0 or byte_stride == elem_size:
            # Contiguous (no stride or stride equals element size)
            arr = np.frombuffer(buffer_data, dtype=dt, count=count * nc,
                               offset=byte_offset).copy()
        else:
            # Interleaved vertex attributes -read each row with stride
            arr = np.ndarray(shape=(count, nc), dtype=dt,
                             buffer=buffer_data,
                             offset=byte_offset,
                             strides=(byte_stride, dt.itemsize)).copy()
    else:
        arr = np.zeros(count * nc, dtype=dt)
    if nc > 1:
        arr = arr.reshape(count, nc)

    sparse = acc.get('sparse')
    if sparse:
        if not isinstance(sparse, dict):
            raise ValueError("Invalid sparse accessor object")
        sparse_count = _safe_int(sparse.get('count'), 0)
        if sparse_count < 0:
            raise ValueError(f"Invalid sparse accessor count: {sparse.get('count')}")
        indices_info = sparse.get('indices', {})
        values_info = sparse.get('values', {})
        if indices_info.get('bufferView') is None or values_info.get('bufferView') is None:
            raise ValueError("Sparse accessor missing bufferView")
        buffer_views = gltf.get('bufferViews', [])
        index_bv_idx = indices_info.get('bufferView')
        value_bv_idx = values_info.get('bufferView')
        if not isinstance(index_bv_idx, int) or index_bv_idx < 0 or index_bv_idx >= len(buffer_views):
            raise ValueError(f"Invalid sparse index bufferView: {index_bv_idx}")
        if not isinstance(value_bv_idx, int) or value_bv_idx < 0 or value_bv_idx >= len(buffer_views):
            raise ValueError(f"Invalid sparse value bufferView: {value_bv_idx}")
        index_bv = buffer_views[index_bv_idx]
        if indices_info.get('componentType') not in (5121, 5123, 5125):
            raise ValueError(f"Unsupported sparse index componentType: {indices_info.get('componentType')}")
        index_dt = np.dtype(_DTYPE_MAP[indices_info['componentType']]).newbyteorder('<')
        index_offset = _safe_int(index_bv.get('byteOffset'), 0) + _safe_int(indices_info.get('byteOffset'), 0)
        index_required = sparse_count * index_dt.itemsize
        index_buffer = _buffer_data(bin_data, index_bv.get('buffer', 0))
        if index_buffer is None or index_offset < 0 or index_offset + index_required > len(index_buffer):
            raise ValueError("Sparse index buffer range exceeds buffer data")
        sparse_indices = np.frombuffer(
            index_buffer, dtype=index_dt, count=sparse_count, offset=index_offset
        ).astype(np.uint32)
        if sparse_indices.size and int(sparse_indices.max()) >= count:
            raise ValueError("Sparse accessor index out of range")

        value_bv = buffer_views[value_bv_idx]
        value_offset = _safe_int(value_bv.get('byteOffset'), 0) + _safe_int(values_info.get('byteOffset'), 0)
        value_required = sparse_count * nc * dt.itemsize
        value_buffer = _buffer_data(bin_data, value_bv.get('buffer', 0))
        if value_buffer is None or value_offset < 0 or value_offset + value_required > len(value_buffer):
            raise ValueError("Sparse value buffer range exceeds buffer data")
        sparse_values = np.frombuffer(
            value_buffer, dtype=dt, count=sparse_count * nc, offset=value_offset
        ).copy()
        if nc > 1:
            sparse_values = sparse_values.reshape(sparse_count, nc)
        arr[sparse_indices] = sparse_values

    component_type = acc['componentType']
    if acc.get('normalized', False) and component_type in (5120, 5121, 5122, 5123, 5125):
        arr = arr.astype(np.float32)
        if component_type == 5120:
            arr = np.maximum(arr / 127.0, -1.0)
        elif component_type == 5121:
            arr = arr / 255.0
        elif component_type == 5122:
            arr = np.maximum(arr / 32767.0, -1.0)
        elif component_type == 5123:
            arr = arr / 65535.0
        elif component_type == 5125:
            arr = arr / 4294967295.0
    elif component_type in (5121, 5123, 5125):
        arr = arr.astype(np.uint32)
    elif component_type == 5126:
        arr = arr.astype(np.float32)
    return arr


__all__ = ["_coerce_vec_array", "_get_accessor"]
