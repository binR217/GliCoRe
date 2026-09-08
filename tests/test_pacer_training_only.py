import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    ROOT / "glicore/network_architecture/tumor/glicore_tumor.py",
    ROOT / "glicore/network_architecture/glicore_generalized.py",
)


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def test_standard_forward_accepts_only_input_tensor():
    for path, class_name in zip(MODEL_FILES, ("GliCoRe", "GliCoReGeneralized")):
        cls = _class_node(path, class_name)
        forward = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        assert [argument.arg for argument in forward.args.args] == ["self", "x"]


def test_public_models_expose_no_pacer_inference_state():
    forbidden = ("PACER_INFERENCE_MODE", "pacer_inference_mode", "tiled_inference_context")
    for path in MODEL_FILES:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden)


def test_standard_forward_explicitly_disables_pacer():
    for path in MODEL_FILES:
        text = path.read_text(encoding="utf-8")
        assert "use_pacer=False" in text
