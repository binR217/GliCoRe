from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_release_roots_exist():
    required = {
        "glicore",
        "training_scripts",
        "tests",
        "README.md",
        "requirements.txt",
        "LICENSE",
        ".gitignore",
    }
    assert required.issubset({path.name for path in ROOT.iterdir()})


def test_release_contains_no_binary_artifacts():
    forbidden_suffixes = {
        ".pth",
        ".pt",
        ".ckpt",
        ".nii",
        ".nii.gz",
        ".npz",
        ".npy",
        ".pkl",
        ".pyc",
    }
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in forbidden_suffixes):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
