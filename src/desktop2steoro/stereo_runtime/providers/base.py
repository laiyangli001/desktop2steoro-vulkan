from __future__ import annotations

from typing import Protocol

import torch

from stereo_runtime.depth_provider import DepthProfileResult


class DepthProviderProtocol(Protocol):
    def load(self):
        ...

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        ...

