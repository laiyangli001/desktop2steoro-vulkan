from __future__ import annotations

import numpy as np
import pytest

from stereo_runtime.output_convert import runtime_output_to_numpy


def test_runtime_output_to_numpy_keeps_hwc_uint8_rgb():
    frame = np.zeros((6, 5, 4), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30
    frame[..., 3] = 255

    result = runtime_output_to_numpy(frame)

    assert result.shape == (6, 5, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [10, 20, 30]


def test_runtime_output_to_numpy_converts_chw_float():
    frame = np.array(
        [
            [[0.0, 0.5], [1.0, 2.0]],
            [[0.25, 0.5], [0.75, 1.0]],
            [[-1.0, 0.0], [0.5, 1.0]],
        ],
        dtype=np.float32,
    )

    result = runtime_output_to_numpy(frame)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [0, 63, 0]
    assert result[1, 1].tolist() == [255, 255, 255]


def test_runtime_output_to_numpy_uses_first_batch_item():
    frame = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    frame[0, 0, :, :] = 7

    result = runtime_output_to_numpy(frame)

    assert result.shape == (4, 5, 3)
    assert result[..., 0].max() == 7


def test_runtime_output_to_numpy_rejects_unknown_shape():
    with pytest.raises(RuntimeError, match="Unsupported runtime output shape"):
        runtime_output_to_numpy(np.zeros((4, 5), dtype=np.uint8))
