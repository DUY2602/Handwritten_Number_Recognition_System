from __future__ import annotations

from typing import Any, Dict, List

from segmentation.context_corrector import (
    build_raw_predictions,
    build_raw_predictions_with_top_k,
)
from segmentation.expression_parser import build_and_evaluate
from segmentation.prediction_refiner import refine_predictions_by_line
from segmentation.segmentation import segment_image


LINE_GROUP_ERROR = "The detected characters could not be grouped into a valid expression line."
NO_CHARACTERS_ERROR = "No characters detected."


def summarize_line_predictions(line_predictions) -> Dict[str, Any]:
    lines: List[Dict[str, Any]] = []

    for default_index, line in enumerate(line_predictions or []):
        characters = list(line.get("characters") or [])
        raw_chars = [item["char"] for item in characters]
        expression_str, result_str, error = build_and_evaluate(raw_chars)

        lines.append({
            "line_index": int(line.get("line_index", default_index)),
            "expression": expression_str,
            "result": result_str,
            "error": error,
            "characters": characters,
            "rect": tuple(int(value) for value in line.get("rect", (0, 0, 0, 0))),
        })

    if len(lines) == 1:
        expression = lines[0]["expression"]
        result = lines[0]["result"]
        error = lines[0]["error"]
    else:
        success_count = sum(1 for item in lines if not item["error"])
        expression = f"Detected {len(lines)} lines"
        result = f"{success_count}/{len(lines)} lines evaluated successfully"
        error = None

    return {
        "lines": lines,
        "expression": expression,
        "result": result,
        "error": error,
    }


def analyze_expression_image(
    image_path,
    input_mode="upload",
    include_model_top_k: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    roi_images, rects, thresh, img_display = segment_image(
        image_path,
        input_mode=input_mode,
    )

    analysis: Dict[str, Any] = {
        "roi_images": roi_images,
        "rects": rects,
        "thresh": thresh,
        "img_display": img_display,
        "raw_predictions": [],
        "line_predictions": [],
        "lines": [],
        "expression": "",
        "result": None,
        "error": None,
        "fatal_error": None,
    }

    if not roi_images:
        analysis["fatal_error"] = NO_CHARACTERS_ERROR
        return analysis

    if include_model_top_k:
        raw_predictions = build_raw_predictions_with_top_k(
            roi_images,
            top_k=top_k,
        )
    else:
        raw_predictions = build_raw_predictions(roi_images)

    line_predictions = refine_predictions_by_line(rects, roi_images, raw_predictions)

    analysis["raw_predictions"] = raw_predictions
    analysis["line_predictions"] = line_predictions

    if not line_predictions:
        analysis["fatal_error"] = LINE_GROUP_ERROR
        return analysis

    summary = summarize_line_predictions(line_predictions)
    analysis.update(summary)
    return analysis
