from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_evidential_transform_matches_the_paper():
    tumor = _text("glicore/network_architecture/tumor/mefc.py")
    transfer = _text("glicore/network_architecture/glicore_generalized.py")
    for text in (tumor, transfer):
        assert "F.softplus" in text
        assert "+ 1e-2" in text
        assert "max=5.0" in text
        assert "+ 1e-4" in text


def test_pacer_relation_settings_match_the_paper():
    config = _text("glicore/glicore_config.py")
    loss = _text("glicore/training/loss_functions/pacer_loss.py")
    for token in ("relation_margin: float = 0.3", "max_adjacent_pairs: int = 2048",
                  "max_axis_spacing_mm: float = 3.0"):
        assert token in config
    assert "cosine_similarity" in loss


def test_standard_inference_disables_pacer():
    for relative in (
        "glicore/network_architecture/tumor/glicore_tumor.py",
        "glicore/network_architecture/glicore_generalized.py",
    ):
        assert "use_pacer=False" in _text(relative)


def test_all_reported_dataset_regions_are_evaluated():
    evaluator = _text("glicore/evaluation/evaluate_predictions.py")
    for token in ("WT", "ET", "TC", "Spl", "RKid", "Pan", "RV", "Myo", "LV"):
        assert '"%s"' % token in evaluator
