"""Repository diagnostic tools exposed as importable Python modules."""

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_APP_ROOT = _SRC_ROOT / "desktop2stereo"
for _import_root in (_SRC_ROOT, _APP_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
