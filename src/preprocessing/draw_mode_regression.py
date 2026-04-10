from __future__ import annotations


from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import os
import sys
import tempfile

import cv2
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent # src/model/evaluation
MODEL_DIR = CURRENT_DIR.parent # src/model
SRC_DIR = MODEL_DIR.parent # src
PROJECT_ROOT = SRC_DIR.parent # project root
if str(SRC_DIR) not in sys.path: # Ensure src is in path for absolute imports
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class DrawModeCase:
    name: str
    expression: str
    paren_style: str = "round"
    note: str = ""


@dataclass(frozen=True)
class SavedDrawCase:
    name: str
    analysis_id: str
    expected_expression: str
    note: str = ""


DEFAULT_DRAW_MODE_CASES: List[DrawModeCase] = [
    DrawModeCase(
        name="open_paren_round_basic",
        expression="(7-3)",
        paren_style="round",
        note="Baseline open parenthesis at the start of the line.",
    ),
    DrawModeCase(
        name="open_paren_squareish_basic",
        expression="(7-3)",
        paren_style="squareish",
        note="Bracket-like opening stroke, close to the user draw style.",
    ),
    DrawModeCase(
        name="open_paren_squareish_chain",
        expression="(7-6+5)*3/7",
        paren_style="squareish",
        note="Opening bracket plus multiple operators on one line.",
    ),
    DrawModeCase(
        name="inner_paren_multi_digit",
        expression="21*(50+2)",
        paren_style="round",
        note="Tests multi-digit groups and inner opening parenthesis.",
    ),
    DrawModeCase(
        name="operator_then_open_paren",
        expression="1+(2*3)",
        paren_style="round",
        note="Opening parenthesis appears after an operator.",
    ),
    DrawModeCase(
        name="slash_then_open_paren",
        expression="9/(3+6)",
        paren_style="squareish",
        note="Opening parenthesis immediately after slash.",
    ),
    DrawModeCase(
        name="two_group_product",
        expression="(4+5)*(6-2)",
        paren_style="round",
        note="Two parenthesized groups with multiplication.",
    ),
]


DEFAULT_SAVED_DRAW_CASES: List[SavedDrawCase] = [
    SavedDrawCase(
        name="real_draw_open_paren_basic",
        analysis_id="20260408T053056_b74d5332",
        expected_expression="(7-3)",
        note="Short real draw sample with opening parenthesis at the start.",
    ),
    SavedDrawCase(
        name="real_draw_open_paren_then_multiply",
        analysis_id="20260408T053107_d896e4df",
        expected_expression="(6*5)/7",
        note="Real draw sample with opening parenthesis and slash.",
    ),
    SavedDrawCase(
        name="real_draw_open_paren_chain",
        analysis_id="20260408T053255_681e9320",
        expected_expression="(7-6+5)*3/7",
        note="Real draw chain that used to confuse opening parenthesis with 1.",
    ),
    SavedDrawCase(
        name="real_draw_open_paren_multistep",
        analysis_id="20260408T053154_be0bc558",
        expected_expression="(7-6)*5/7",
        note="Real draw sample after the opening parenthesis refinement tweak.",
    ),
    SavedDrawCase(
        name="real_draw_open_paren_with_leading_digit",
        analysis_id="20260408T053350_62705004",
        expected_expression="(6-5+1)+5/7",
        note="Real draw sample where the first ROI looked like 1 but should open a group.",
    ),
    SavedDrawCase(
        name="real_draw_inner_paren_multi_digit",
        analysis_id="20260408T054136_b3f9e4ca",
        expected_expression="21*(50+2)",
        note="Real draw sample with multi-digit numbers and inner parentheses.",
    ),
]


def _compute_expected_result(expression: str) -> str:
    from segmentation.expression_parser import build_and_evaluate # expression_parser is still in src/segmentation

    expr, result, error = build_and_evaluate(list(expression))
    if error:
        raise ValueError(f"Invalid benchmark expression {expression!r}: {error}")
    if expr != expression:
        raise ValueError(
            f"Benchmark expression normalized unexpectedly: expected {expression!r}, got {expr!r}."
        )
    return str(result)


def _digit_text_metrics(char: str, font_scale: float, thickness: int) -> tuple[tuple[int, int], int]:
    return cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)


def _char_advance(char: str, font_scale: float, thickness: int) -> int:
    if char.isdigit():
        (width, _), _ = _digit_text_metrics(char, font_scale, thickness)
        return width + 26
    return {
        "+": 84,
        "-": 88,
        "*": 82,
        "/": 76,
        "(": 74,
        ")": 74,
    }.get(char, 70)


def _draw_plus(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 0, thickness)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), 0, thickness)


def _draw_minus(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 0, thickness)


def _draw_multiply(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    arm = size // 2
    cv2.line(img, (cx - arm, cy - arm), (cx + arm, cy + arm), 0, thickness)
    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), 0, thickness)


def _draw_divide(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    arm = size // 2
    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), 0, thickness)


def _draw_left_round_paren(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    axes = (max(4, size // 6), max(12, size // 2))
    cv2.ellipse(img, (cx + 2, cy), axes, 0, 78, 282, 0, thickness)


def _draw_right_round_paren(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    axes = (max(4, size // 6), max(12, size // 2))
    cv2.ellipse(img, (cx - 2, cy), axes, 0, -102, 102, 0, thickness)


def _draw_left_squareish_paren(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    half_h = max(18, size // 2)
    arm = max(10, size // 4)
    x = cx - max(4, arm // 3)
    y0 = cy - half_h
    y1 = cy + half_h
    cv2.line(img, (x + arm, y0), (x, y0), 0, thickness)
    cv2.line(img, (x, y0), (x, y1), 0, thickness)
    cv2.line(img, (x, y1), (x + arm, y1), 0, thickness)


def _draw_right_squareish_paren(img: np.ndarray, cx: int, cy: int, size: int, thickness: int) -> None:
    half_h = max(18, size // 2)
    arm = max(10, size // 4)
    x = cx + max(4, arm // 3)
    y0 = cy - half_h
    y1 = cy + half_h
    cv2.line(img, (x - arm, y0), (x, y0), 0, thickness)
    cv2.line(img, (x, y0), (x, y1), 0, thickness)
    cv2.line(img, (x - arm, y1), (x, y1), 0, thickness)


def _draw_symbol(
    canvas: np.ndarray,
    char: str,
    x: int,
    center_y: int,
    style: str,
    index: int,
) -> None:
    y_jitter = (-4, 2, -1, 3, 0)[index % 5]
    cx = x + (_char_advance(char, 3.0, 9) // 2)
    cy = center_y + y_jitter

    if char == "+":
        _draw_plus(canvas, cx, cy, size=28, thickness=8)
        return
    if char == "-":
        _draw_minus(canvas, cx, cy, size=36, thickness=8)
        return
    if char == "*":
        _draw_multiply(canvas, cx, cy, size=24, thickness=7)
        return
    if char == "/":
        _draw_divide(canvas, cx, cy, size=30, thickness=8)
        return
    if char == "(":
        if style == "squareish":
            _draw_left_squareish_paren(canvas, cx, cy, size=118, thickness=8)
        else:
            _draw_left_round_paren(canvas, cx, cy, size=118, thickness=8)
        return
    if char == ")":
        if style == "squareish":
            _draw_right_squareish_paren(canvas, cx, cy, size=118, thickness=8)
        else:
            _draw_right_round_paren(canvas, cx, cy, size=118, thickness=8)
        return

    raise ValueError(f"Unsupported symbol: {char}")


def _load_evaluation_module():
    module_name = "draw_mode_regression_eval_module"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = CURRENT_DIR / "evaluation.py" # evaluation.py is now in the same directory
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def render_draw_expression(
    expression: str,
    output_path: str | Path,
    *,
    paren_style: str = "round",
    canvas_height: int = 320,
) -> Path:
    font_scale = 3.0
    thickness = 9
    padding_x = 34
    center_y = canvas_height // 2
    baseline_y = center_y + 48

    total_width = (padding_x * 2) + sum(
        _char_advance(char, font_scale, thickness)
        for char in expression
    )
    canvas = np.full((canvas_height, max(640, total_width)), 255, dtype=np.uint8)

    x = padding_x
    for index, char in enumerate(expression):
        if char.isdigit():
            y_jitter = (-4, 0, 3, -2, 1)[index % 5]
            cv2.putText(
                canvas,
                char,
                (x, baseline_y + y_jitter),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                0,
                thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            _draw_symbol(canvas, char, x, center_y, paren_style, index)
        x += _char_advance(char, font_scale, thickness)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Could not write rendered draw image to: {output}")
    return output


def evaluate_draw_mode_cases(
    cases: Sequence[DrawModeCase] = DEFAULT_DRAW_MODE_CASES,
    *,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    evaluation_module = _load_evaluation_module()

    created_temp_dir = None
    if output_dir is None:
        created_temp_dir = tempfile.TemporaryDirectory(prefix="draw_mode_regression_")
        output_root = Path(created_temp_dir.name)
    else:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

    results = []
    expr_correct = 0
    result_correct = 0

    try:
        for case in cases:
            image_path = output_root / f"{case.name}.png"
            render_draw_expression(case.expression, image_path, paren_style=case.paren_style)
            expected_result = _compute_expected_result(case.expression)
            debug = evaluation_module.evaluate_full_pipeline_debug(
                str(image_path),
                case.expression,
                input_mode="draw",
            )

            expr_match = debug["pred_expr"] == case.expression
            result_match = debug["pred_result"] == expected_result
            expr_correct += int(expr_match)
            result_correct += int(result_match)

            results.append(
                {
                    **asdict(case),
                    "image_path": str(image_path),
                    "expected_result": expected_result,
                    "pred_expr": debug["pred_expr"],
                    "pred_result": debug["pred_result"],
                    "pred_error": debug["pred_error"],
                    "num_rois": debug["num_rois"],
                    "segmentation_status": debug["segmentation"]["status"],
                    "segmentation_issue": debug["segmentation"]["issue"],
                    "expr_match": expr_match,
                    "result_match": result_match,
                }
            )
    finally:
        if created_temp_dir is not None:
            created_temp_dir.cleanup()

    total = len(results)
    return {
        "total_cases": total,
        "expression_accuracy": (expr_correct / total) if total else 0.0,
        "result_accuracy": (result_correct / total) if total else 0.0,
        "expression_correct": expr_correct,
        "result_correct": result_correct,
        "cases": results,
        "output_dir": str(output_root),
    }


def _analysis_root() -> Path:
    return SRC_DIR / "model" / "artifacts" / "feedback" / "analyses"


def evaluate_saved_draw_cases(
    cases: Sequence[SavedDrawCase] = DEFAULT_SAVED_DRAW_CASES,
) -> Dict[str, Any]:
    from segmentation.expression_parser import build_and_evaluate # expression_parser is still in src/segmentation
    from segmentation.prediction_refiner import refine_predictions_by_line # prediction_refiner is still in src/segmentation

    results = []
    expr_correct = 0
    result_correct = 0

    for case in cases:
        analysis_path = _analysis_root() / case.analysis_id / "analysis.json"
        if not analysis_path.exists(): # _analysis_root() is updated below
            raise FileNotFoundError(f"Missing analysis record: {analysis_path}")

        import json

        with analysis_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)

        rects = []
        roi_images = []
        raw_predictions = []
        for item in record.get("characters", []):
            rects.append(tuple(int(v) for v in item["rect"]))
            roi = cv2.imread(str(item["roi_path"]), cv2.IMREAD_GRAYSCALE)
            if roi is None:
                raise FileNotFoundError(f"Missing ROI image: {item['roi_path']}")
            roi_images.append(roi)
            raw_predictions.append(
                {
                    "char": str(item.get("raw_char", item["char"])),
                    "conf": float(item.get("raw_conf", item["conf"])),
                    "raw_char": str(item.get("raw_char", item["char"])),
                    "raw_conf": float(item.get("raw_conf", item["conf"])),
                    "top_k": list(item.get("top_k", [])),
                }
            )

        line_predictions = refine_predictions_by_line(rects, roi_images, raw_predictions)
        if len(line_predictions) == 1:
            pred_chars = [entry["char"] for entry in line_predictions[0]["characters"]]
            pred_expr, pred_result, pred_error = build_and_evaluate(pred_chars)
        else:
            pred_expr = "Detected multiple lines"
            pred_result = None
            pred_error = "Expected a single expression line."

        expected_result = _compute_expected_result(case.expected_expression)
        expr_match = pred_expr == case.expected_expression
        result_match = pred_result == expected_result
        expr_correct += int(expr_match)
        result_correct += int(result_match)

        results.append(
            {
                **asdict(case),
                "pred_expr": pred_expr,
                "pred_result": pred_result,
                "pred_error": pred_error,
                "expected_result": expected_result,
                "expr_match": expr_match,
                "result_match": result_match,
                "raw_expression": str(record.get("expression", "")),
            }
        )

    total = len(results)
    return {
        "total_cases": total,
        "expression_accuracy": (expr_correct / total) if total else 0.0,
        "result_accuracy": (result_correct / total) if total else 0.0,
        "expression_correct": expr_correct,
        "result_correct": result_correct,
        "cases": results,
    }


def format_draw_mode_report(summary: Dict[str, Any]) -> str:
    lines = [
        "Draw mode regression",
        f"- Cases: {summary['total_cases']}",
        (
            f"- Expression accuracy: {summary['expression_correct']}/{summary['total_cases']} "
            f"= {summary['expression_accuracy'] * 100:.1f}%"
        ),
        (
            f"- Result accuracy: {summary['result_correct']}/{summary['total_cases']} "
            f"= {summary['result_accuracy'] * 100:.1f}%"
        ),
        "",
        "Per-case details",
    ]

    for item in summary["cases"]:
        lines.append(
            "- "
            f"{item['name']}: "
            f"expected={item['expression']!r} -> {item['expected_result']!r}, "
            f"pred={item['pred_expr']!r} -> {item['pred_result']!r}, "
            f"expr_ok={item['expr_match']}, result_ok={item['result_match']}, "
            f"seg={item['segmentation_status']}"
        )
        if item["pred_error"]:
            lines.append(f"  error={item['pred_error']}")
        if item["segmentation_issue"]:
            lines.append(f"  seg_issue={item['segmentation_issue']}")

    return "\n".join(lines)


def format_saved_draw_report(summary: Dict[str, Any]) -> str:
    lines = [
        "Saved draw regression",
        f"- Cases: {summary['total_cases']}",
        (
            f"- Expression accuracy: {summary['expression_correct']}/{summary['total_cases']} "
            f"= {summary['expression_accuracy'] * 100:.1f}%"
        ),
        (
            f"- Result accuracy: {summary['result_correct']}/{summary['total_cases']} "
            f"= {summary['result_accuracy'] * 100:.1f}%"
        ),
        "",
        "Per-case details",
    ]

    for item in summary["cases"]:
        lines.append(
            "- "
            f"{item['name']}: "
            f"expected={item['expected_expression']!r} -> {item['expected_result']!r}, "
            f"pred={item['pred_expr']!r} -> {item['pred_result']!r}, "
            f"expr_ok={item['expr_match']}, result_ok={item['result_match']}"
        )
        if item["pred_error"]:
            lines.append(f"  error={item['pred_error']}")

    return "\n".join(lines)


def iter_default_case_expressions(cases: Iterable[DrawModeCase] = DEFAULT_DRAW_MODE_CASES) -> List[str]:
    return [case.expression for case in cases]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    synthetic_report = evaluate_draw_mode_cases( # This path is relative to the project root
        output_dir=CURRENT_DIR / ".." / ".." / ".tmp" / "draw_mode_regression"
    )
    print(format_draw_mode_report(synthetic_report))
    print("")
    saved_report = evaluate_saved_draw_cases()
    print(format_saved_draw_report(saved_report))
