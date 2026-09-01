"""Compatibility module entry point with the shared authorization gate."""

try:
    from desktop2stereo.app_runtime.bootstrap import main
except ModuleNotFoundError:
    from app_runtime.bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main(["--gui2"]))
