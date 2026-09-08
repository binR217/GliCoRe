from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json"}
FORBIDDEN = (
    "CG" + "LA",
    "cg" + "la",
    "Ours" + "16",
    "ours" + "16",
    "AD" + "FPA",
    "ad" + "fpa",
    "Pro" + " Max",
    "神" + "级",
    "内" + "鬼",
)


def test_historical_names_are_absent():
    offenders = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            relative = path.relative_to(ROOT).as_posix()
            if any(token in text or token in relative for token in FORBIDDEN):
                offenders.append(relative)
    assert offenders == []


def test_canonical_release_files_exist():
    required = {
        "glicore/network_architecture/glicore_generalized.py",
        "glicore/network_architecture/tumor/glicore_tumor.py",
        "glicore/training/loss_functions/glicore_evidential_loss.py",
        "glicore/training/network_training/pacer_training_support.py",
    }
    existing = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
    }
    assert required.issubset(existing)
