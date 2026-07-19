"""Validate the project requirements traceability matrix."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs" / "requirements-matrix.md"
VALID_STATUSES = {"planned", "in_progress", "implemented", "verified", "accepted"}
REQUIRED_COLUMNS = ("ID", "领域", "必须遵循的要求", "规范来源", "代码映射", "测试/验收", "状态")
COMPLETED_STATUSES = {"verified", "accepted"}


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def load_requirements() -> list[dict[str, str]]:
    lines = MATRIX.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("| ID |")
    )
    headers = _cells(lines[header_index])
    if tuple(headers) != REQUIRED_COLUMNS:
        raise ValueError(f"unexpected matrix columns: {headers}")

    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("| "):
            if rows:
                break
            continue
        values = _cells(line)
        if len(values) != len(headers) or not re.fullmatch(r"[A-Z]+-\d{3}", values[0]):
            continue
        rows.append(dict(zip(headers, values)))
    return rows


def _validate_path(value: str, *, row_id: str, label: str, strict: bool, errors: list[str]) -> None:
    if value == "TBD":
        if strict:
            errors.append(f"{row_id}: {label} is TBD in strict mode")
        return
    for raw_path in value.split(";"):
        path = raw_path.strip().strip("`")
        if (
            not path
            or path.startswith(("实机", "CI", "Actions"))
            or "/" not in path
            or "\\" in path
            or path.endswith(("测试", "验收", "profiling", "benchmark", "runner", "audit"))
        ):
            continue
        if not (ROOT / path).exists():
            errors.append(f"{row_id}: {label} path does not exist: {path}")


def validate(*, strict: bool = False) -> list[str]:
    errors: list[str] = []
    if not MATRIX.exists():
        return [f"missing matrix: {MATRIX}"]
    rows = load_requirements()
    if not rows:
        return ["requirements matrix has no data rows"]

    ids = [row["ID"] for row in rows]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    errors.extend(f"duplicate requirement id: {item}" for item in duplicates)
    for row in rows:
        row_id = row["ID"]
        for column in REQUIRED_COLUMNS:
            if not row[column]:
                errors.append(f"{row_id}: empty {column}")
        status = row["状态"]
        if status not in VALID_STATUSES:
            errors.append(f"{row_id}: invalid status {status}")
        if strict and status not in COMPLETED_STATUSES:
            errors.append(f"{row_id}: status {status} is not releasable")
        _validate_path(row["代码映射"], row_id=row_id, label="code mapping", strict=strict, errors=errors)
        _validate_path(row["测试/验收"], row_id=row_id, label="test mapping", strict=strict, errors=errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="require every row to be verified or accepted")
    args = parser.parse_args(argv)
    errors = validate(strict=args.strict)
    if errors:
        print("Compliance check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Compliance check passed: {len(load_requirements())} requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
