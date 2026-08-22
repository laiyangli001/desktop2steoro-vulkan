from __future__ import annotations

import numpy as np

from utils.cpu_warnings import describe_tensor, warn_cpu_transfer


def runtime_output_to_numpy(frame):
    import torch

    if isinstance(frame, torch.Tensor):
        frame = frame.detach()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[0] in (3, 4):
            frame = frame[:3].permute(1, 2, 0)
        elif frame.ndim == 3 and frame.shape[-1] >= 3:
            frame = frame[..., :3]
        else:
            raise RuntimeError(f"Unsupported runtime output shape: {tuple(frame.shape)}")
        if frame.is_floating_point():
            frame = frame.clamp(0.0, 1.0).mul(255.0)
        warn_cpu_transfer(
            "runtime output conversion",
            ".cpu().numpy()",
            detail=describe_tensor(frame),
            key="runtime_output_to_numpy_cpu_transfer",
        )
        return frame.contiguous().to(torch.uint8).cpu().numpy()

    warn_cpu_transfer(
        "runtime output conversion",
        "numpy input path",
        detail=f"type={type(frame).__name__}",
        key="runtime_output_to_numpy_numpy_input",
    )
    frame_np = np.asarray(frame)
    if frame_np.ndim == 4:
        frame_np = frame_np[0]
    if frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
        frame_np = np.transpose(frame_np[:3], (1, 2, 0))
    elif frame_np.ndim == 3 and frame_np.shape[-1] >= 3:
        frame_np = frame_np[..., :3]
    else:
        raise RuntimeError(f"Unsupported runtime output shape: {tuple(frame_np.shape)}")
    if np.issubdtype(frame_np.dtype, np.floating):
        frame_np = np.clip(frame_np, 0.0, 1.0) * 255.0
    return frame_np.astype("uint8", copy=False)
