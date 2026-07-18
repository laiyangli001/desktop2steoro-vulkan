# Scripts Layout

This directory keeps runnable helper scripts grouped by purpose.

| Directory | Purpose |
|---|---|
| `benchmark/` | Performance benchmarks and profiling entry points. |
| `examples/` | Small demos and preview generators. |
| `smoke/` | Fast smoke checks for the public API and host integration examples. |
| `tools/` | Model export, depth generation, comparison, and consistency utilities. |
| `windows/` | Visible Windows launcher scripts for manual testing. |
| `dev/` | Low-level development probes for TensorRT/Triton kernels. |

Scripts under subdirectories should resolve the project root with:

```python
ROOT = Path(__file__).resolve().parents[2]
```

Keep root-level `scripts/` empty except for this README and category folders.
