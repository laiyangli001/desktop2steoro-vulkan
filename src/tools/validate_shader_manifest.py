"""Validate shader source declarations against the versioned manifest."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


_WORKGROUP_RE = re.compile(
    r"layout\s*\(\s*local_size_x\s*=\s*(\d+)\s*,\s*"
    r"local_size_y\s*=\s*(\d+)\s*,\s*local_size_z\s*=\s*(\d+)\s*\)\s*in"
)
_ENTRY_RE = re.compile(r"\bvoid\s+(main)\s*\(")
_BINDING_RE = re.compile(
    r"layout\s*\(\s*set\s*=\s*(\d+)\s*,\s*binding\s*=\s*(\d+)"
    r"(?P<qualifiers>[^)]*)\)\s*(?P<body>[^;]+);"
)


def _binding_kind(body: str) -> str:
    if "buffer" in body:
        return "storage_buffer"
    if "image" in body:
        return "storage_image"
    raise ValueError(f"unsupported descriptor declaration: {body.strip()}")


def _binding_access(qualifiers: str, body: str) -> str:
    declaration = f"{qualifiers} {body}"
    if "readonly" in declaration:
        return "read_only"
    if "writeonly" in declaration:
        return "write_only"
    if "buffer" in body or "uniform image" in body:
        return "read_write"
    return "read_write"


def _source_bindings(source: str) -> list[dict[str, object]]:
    bindings = []
    for match in _BINDING_RE.finditer(source):
        qualifiers = match.group("qualifiers")
        body = match.group("body")
        descriptor = {
            "set": int(match.group(1)),
            "binding": int(match.group(2)),
            "kind": _binding_kind(body),
            "access": _binding_access(qualifiers, body),
            "format": next(
                (token for token in ("rgba8",) if token in f"{qualifiers} {body}"), None
            ),
        }
        bindings.append(descriptor)
    return bindings


def validate_manifest(root: Path) -> list[str]:
    manifest_path = root / "shaders" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")

    entries = payload.get("shaders")
    if not isinstance(entries, list) or not entries:
        return ["manifest shaders must be a non-empty list"]

    names: set[str] = set()
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in names:
            errors.append(f"invalid or duplicate shader name: {name!r}")
            continue
        names.add(name)
        source_path = root / "shaders" / entry["source"]
        spirv_path = root / "shaders" / entry["spirv"]
        if not source_path.is_file():
            errors.append(f"{name}: missing source {entry['source']}")
            continue
        if not spirv_path.is_file():
            errors.append(f"{name}: missing SPIR-V {entry['spirv']}")

        source = source_path.read_text(encoding="utf-8")
        workgroup_match = _WORKGROUP_RE.search(source)
        if not workgroup_match:
            errors.append(f"{name}: missing local workgroup declaration")
        else:
            actual_workgroup = [int(value) for value in workgroup_match.groups()]
            if actual_workgroup != entry.get("workgroup"):
                errors.append(
                    f"{name}: workgroup {actual_workgroup} != {entry.get('workgroup')}"
                )
        if not _ENTRY_RE.search(source):
            errors.append(f"{name}: missing main entry")
        if entry.get("entry") != "main":
            errors.append(f"{name}: only main entry is currently supported")
        try:
            actual_bindings = _source_bindings(source)
        except ValueError as error:
            errors.append(f"{name}: {error}")
        else:
            expected_bindings = entry.get("descriptor_bindings")
            if actual_bindings != expected_bindings:
                errors.append(
                    f"{name}: bindings {actual_bindings} != {expected_bindings}"
                )
        if entry.get("push_constants_size") != 0:
            errors.append(f"{name}: push constants are not supported by this validator")
        if entry.get("precision") not in {"fp16", "fp32"}:
            errors.append(f"{name}: precision must be fp16 or fp32")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    errors = validate_manifest(root)
    if errors:
        for error in errors:
            print(f"shader manifest: {error}", file=sys.stderr)
        return 1
    print("shader manifest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
