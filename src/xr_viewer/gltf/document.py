"""glTF document and buffer loading helpers."""

import base64
import json
import os
import urllib.parse

from pygltflib import GLTF2

def _gltf_to_dict(gltf):
    return json.loads(gltf.to_json())


def _read_glb_chunks(data):
    """Compatibility helper for diagnostics/tests that still pass GLB bytes."""
    gltf = GLTF2().load_from_bytes(data)
    return _gltf_to_dict(gltf), gltf.binary_blob()


def _decode_data_uri(uri):
    try:
        header, payload = uri.split(',', 1)
    except ValueError:
        raise ValueError('Invalid glTF data URI')
    if ';base64' in header:
        return base64.b64decode(payload)
    return urllib.parse.unquote_to_bytes(payload)


def _load_gltf_document(path):
    gltf_obj = GLTF2().load(path)
    gltf = _gltf_to_dict(gltf_obj)
    base_dir = os.path.dirname(os.path.abspath(path))
    binary_blob = gltf_obj.binary_blob()
    buffers = []
    for index, buf in enumerate(gltf.get('buffers') or []):
        uri = buf.get('uri') if isinstance(buf, dict) else None
        data = None
        if uri:
            if uri.startswith('data:'):
                data = _decode_data_uri(uri)
            else:
                parsed = urllib.parse.urlparse(uri)
                if parsed.scheme not in ('', 'file'):
                    raise ValueError(f'Unsupported glTF buffer URI scheme: {parsed.scheme}')
                rel_path = urllib.parse.unquote(parsed.path if parsed.scheme == 'file' else uri)
                rel_path = rel_path.replace('/', os.sep)
                buffer_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
                with open(buffer_path, 'rb') as bf:
                    data = bf.read()
        elif index == 0 and binary_blob is not None:
            data = binary_blob
        else:
            data = b''
        buffers.append(data)
    return gltf, buffers


def _buffer_data(buffers, buffer_index=0):
    if isinstance(buffers, (bytes, bytearray, memoryview)):
        return buffers if int(buffer_index or 0) == 0 else None
    if not isinstance(buffers, (list, tuple)):
        return None
    try:
        index = int(buffer_index or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if index < 0 or index >= len(buffers):
        return None
    return buffers[index]


__all__ = [
    "_buffer_data",
    "_decode_data_uri",
    "_gltf_to_dict",
    "_load_gltf_document",
    "_read_glb_chunks",
]
