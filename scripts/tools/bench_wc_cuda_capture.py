from __future__ import annotations

import argparse
import inspect
import threading
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["wc_cuda", "wc_rocm"], default="wc_cuda")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--reuse-output-buffer", action="store_true")
    parser.add_argument("--output-buffer-count", type=int, default=6)
    parser.add_argument("--clone", action="store_true")
    args = parser.parse_args()

    capture_module = __import__(args.tool)
    kwargs = {
        "monitor_index": args.monitor,
        "reuse_output_buffer": args.reuse_output_buffer,
        "output_buffer_count": args.output_buffer_count,
    }
    supported = inspect.signature(capture_module.WindowsCapture).parameters
    cap = capture_module.WindowsCapture(**{key: value for key, value in kwargs.items() if key in supported})
    count = 0
    first_ts = None
    last_ts = None

    @cap.event
    def on_frame_arrived(frame, control):
        nonlocal count, first_ts, last_ts
        now = time.perf_counter()
        if first_ts is None:
            first_ts = now
        last_ts = now
        if args.clone:
            frame.frame_buffer.clone()
        count += 1

    @cap.event
    def on_closed():
        pass

    timer = threading.Timer(args.seconds, cap.stop)
    timer.start()
    try:
        cap.start()
    finally:
        timer.cancel()

    elapsed = max((last_ts or time.perf_counter()) - (first_ts or time.perf_counter()), 1e-6)
    print(
        f"direct_wc_cuda monitor={args.monitor} frames={count} "
        f"tool={args.tool} "
        f"elapsed={elapsed:.3f}s fps={count / elapsed:.1f} "
        f"reuse_output_buffer={args.reuse_output_buffer} "
        f"output_buffer_count={args.output_buffer_count} clone={args.clone}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
