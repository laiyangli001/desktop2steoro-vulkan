import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
ALLOW = {
    ROOT / "src" / "depth.py",
}
SKIP_PARTS = {"python3", "python-cu13", "__pycache__"}


def test_no_new_imports_from_legacy_depth_module():
    offenders: list[str] = []
    for base in SCAN_DIRS:
        for path in base.rglob("*.py"):
            if path in ALLOW:
                continue
            if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "depth":
                            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} import depth")
                elif isinstance(node, ast.ImportFrom) and node.module == "depth":
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} from depth import ...")

    assert offenders == []
