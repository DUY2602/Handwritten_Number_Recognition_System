import importlib.util
from pathlib import Path
import sys

import cv2


def _load_draw_mode_regression():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "evaluation" / "draw_mode_regression.py"
    spec = importlib.util.spec_from_file_location("draw_mode_regression_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_draw_mode_cases_are_valid():
    draw_mode_regression = _load_draw_mode_regression()

    expressions = draw_mode_regression.iter_default_case_expressions()
    assert len(expressions) >= 6

    for expr in expressions:
        expected_result = draw_mode_regression._compute_expected_result(expr)
        assert expected_result is not None


def test_render_draw_expression_creates_non_empty_image():
    draw_mode_regression = _load_draw_mode_regression()
    root = Path(__file__).resolve().parents[1]
    tmp_dir = root / ".tmp" / "test_draw_mode_regression"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / "draw_case.png"
    draw_mode_regression.render_draw_expression(
        "(7-6+5)*3/7",
        output_path,
        paren_style="squareish",
    )

    image = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert image.shape[0] >= 200
    assert image.shape[1] >= 400
    assert int((image < 250).sum()) > 0


def test_saved_draw_cases_exist():
    draw_mode_regression = _load_draw_mode_regression()
    analysis_root = draw_mode_regression._analysis_root()

    for case in draw_mode_regression.DEFAULT_SAVED_DRAW_CASES:
        analysis_path = analysis_root / case.analysis_id / "analysis.json"
        assert analysis_path.exists(), f"Missing saved draw analysis: {analysis_path}"
