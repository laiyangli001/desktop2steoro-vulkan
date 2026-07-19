from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.check_compliance import load_requirements, validate


def test_requirements_matrix_is_complete() -> None:
    rows = load_requirements()
    assert len(rows) >= 40
    assert len({row["ID"] for row in rows}) == len(rows)
    assert all(row["规范来源"] for row in rows)
    assert all(
        row["代码映射"] != "TBD"
        for row in rows
        if row["状态"] in {"implemented", "verified", "accepted"}
    )


def test_requirements_matrix_has_no_structural_errors() -> None:
    assert validate() == []
