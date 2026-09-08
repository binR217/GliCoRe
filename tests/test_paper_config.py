import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "glicore/glicore_config.py"


def _assignments():
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_paper_config_exists():
    assert CONFIG.is_file()


def test_paper_constants_are_present():
    text = CONFIG.read_text(encoding="utf-8")
    for value in ("1000", "0.99", "3e-5", "1e-2", "0.9", "2048", "3.0"):
        assert value in text


def test_dataset_profiles_are_present():
    text = CONFIG.read_text(encoding="utf-8")
    for value in ("brats", "synapse", "acdc", "et_wt", "foreground_interface"):
        assert value in text


def test_transfer_models_do_not_add_an_unreported_context_cap():
    generalized = (
        ROOT / "glicore/network_architecture/glicore_generalized.py"
    ).read_text(encoding="utf-8")
    synapse = (
        ROOT / "glicore/network_architecture/synapse/glicore_synapse.py"
    ).read_text(encoding="utf-8")
    assert "pacer_max_context_ratio" not in generalized
    assert "pacer_max_context_ratio" not in synapse


def test_brats_calibration_uses_only_the_reported_minibatches():
    text = (
        ROOT / "glicore/training/network_training/glicore_trainer_brats.py"
    ).read_text(encoding="utf-8")
    assert "recheck" not in text.lower()
    assert "pacer_calibration_failed" not in text
