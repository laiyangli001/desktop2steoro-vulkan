from __future__ import annotations

import sys
from pathlib import Path

from path_config import APP_ROOT


ROOT = Path(__file__).resolve().parents[1]
SRC = APP_ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.check_compliance import _validate_path, load_requirements, validate


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


def test_cross_repository_evidence_is_explicitly_scoped() -> None:
    errors: list[str] = []
    _validate_path(
        "../desktop2stereo-site/model/d2s_licensing.go",
        row_id="AUTH-TEST",
        label="code mapping",
        strict=False,
        errors=errors,
    )
    assert errors == []

    _validate_path(
        "../unregistered-repository/model/example.go",
        row_id="AUTH-TEST",
        label="code mapping",
        strict=False,
        errors=errors,
    )
    assert errors == [
        "AUTH-TEST: code mapping path does not exist: ../unregistered-repository/model/example.go"
    ]
