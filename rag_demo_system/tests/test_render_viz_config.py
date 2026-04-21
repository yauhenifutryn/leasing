from __future__ import annotations

import ast
from pathlib import Path


def _load_assignments(path: Path) -> dict[str, object]:
    """Return top-level constant assignments from a Python source file.

    Only recognizes ast.Constant values (int, float, bool, str, None). This is
    deliberately more restrictive than ast.literal_eval to keep the test
    surface minimal and avoid importing the script under test.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue
        if not isinstance(value_node, ast.Constant):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value_node.value
    return values


def test_render_viz_parameters() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_viz.py"
    assert script.exists(), "render_viz.py should exist"
    values = _load_assignments(script)

    assert values.get("PROJECTION_3D") is True
    assert values.get("UMAP_2D_N_NEIGHBORS") == 15
    assert values.get("UMAP_2D_MIN_DIST") == 0.1
    assert values.get("UMAP_3D_N_NEIGHBORS") == 20
    assert values.get("UMAP_3D_MIN_DIST") == 0.2
    assert isinstance(values.get("UMAP_RANDOM_STATE"), int)
