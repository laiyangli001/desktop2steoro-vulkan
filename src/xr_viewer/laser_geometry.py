import numpy as np


def build_laser_beam_vertices(draws, base_half_width, tip_half_width):
    """Build crossed laser beam quads as x,y,z,beam_v float32 vertices.

    Each draw is (start, forward, right, up, length). The beam extends from
    start to start + forward * length. Rendering backends consume this shared
    geometry contract and own their buffer upload and draw calls.
    """
    vertices = []
    for start, forward, right, up, length in draws:
        start = np.asarray(start, dtype=np.float32)
        forward = np.asarray(forward, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        length = max(0.0, float(length))
        end = start + forward * length
        for axis in (right, up):
            axis = np.asarray(axis, dtype=np.float32)
            base_l = start - axis * float(base_half_width)
            base_r = start + axis * float(base_half_width)
            tip_l = end - axis * float(tip_half_width)
            tip_r = end + axis * float(tip_half_width)
            for p, beam_v in (
                (base_l, 0.0), (base_r, 0.0), (tip_l, 1.0),
                (base_r, 0.0), (tip_r, 1.0), (tip_l, 1.0),
            ):
                vertices.extend((float(p[0]), float(p[1]), float(p[2]), float(beam_v)))
    return np.asarray(vertices, dtype=np.float32)
