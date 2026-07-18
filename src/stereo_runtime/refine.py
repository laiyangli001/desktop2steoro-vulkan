from __future__ import annotations

import torch

from .hole_fill import edge_aware_fill


def refine_local(image: torch.Tensor, mask: torch.Tensor | None, enabled: bool = False) -> torch.Tensor:
    if not enabled or mask is None:
        return image
    return edge_aware_fill(image, mask, radius=2, strength=0.5)
