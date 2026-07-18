from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

import torch


@dataclass
class BenchStats:
    timings_ms: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    output_shape: tuple[int, ...] | None = None

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


@contextmanager
def timed(stats: BenchStats, name: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats.timings_ms[name] = (perf_counter() - start) * 1000.0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def read_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
