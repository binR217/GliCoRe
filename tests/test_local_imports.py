import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_absolute_glicore_imports_resolve():
    missing = []
    for path in (ROOT / "glicore").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("glicore"):
                continue
            candidate = ROOT.joinpath(*node.module.split("."))
            if not (candidate.with_suffix(".py").is_file() or
                    (candidate / "__init__.py").is_file()):
                missing.append("%s -> %s" % (path.relative_to(ROOT), node.module))
    assert missing == []
