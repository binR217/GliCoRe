"""Dependency-light runner for the release's static contract tests."""

import importlib.util
import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    failures = []
    count = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and not inspect.signature(function).parameters:
                count += 1
                try:
                    function()
                    print("PASS %s::%s" % (path.name, name))
                except Exception as error:
                    failures.append("%s::%s: %s" % (path.name, name, error))
                    print("FAIL %s::%s" % (path.name, name))
    if failures:
        raise SystemExit("\n".join(failures))
    print("%d static tests passed" % count)


if __name__ == "__main__":
    main()
