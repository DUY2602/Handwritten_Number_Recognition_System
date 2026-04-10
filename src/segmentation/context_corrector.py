"""
Context-aware post-processing for character predictions.

This module keeps the original sequence-level correction helper, and also
exposes a raw prediction builder so multi-line callers can apply correction
after grouping characters by line.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

MULTIPLY_SYMBOL = "\u00d7"
DIVIDE_SYMBOL = "\u00f7"

DIGIT_CHARS = set("0123456789")
OPERATOR_CHARS = {"+", "-", MULTIPLY_SYMBOL, DIVIDE_SYMBOL, "/", "=", "*"}

CORRECTION_CONF_THRESHOLD = 0.82

ALTERNATIVES: Dict[str, List[str]] = {
    "1": ["-"],
    "7": ["/"],
    "0": ["O"],
    "-": ["1"],
    "/": ["7"],
    "+": ["t", "4"],
    MULTIPLY_SYMBOL: ["x", "X", "8"],
    DIVIDE_SYMBOL: ["9"],
    "=": ["="],
    "x": [MULTIPLY_SYMBOL],
    "X": [MULTIPLY_SYMBOL],
}


def _char_type(char: str) -> str:
    if char in DIGIT_CHARS:
        return "digit"
    if char in OPERATOR_CHARS:
        return "operator"
    return "unknown"


def _expected_type(built: List[str]) -> str:
    if not built:
        return "expected_digit"
    if _char_type(built[-1]) == "operator":
        return "expected_digit"
    return "expected_any"


def _is_valid_transition(prev_type: str, next_type: str) -> bool:
    if prev_type == "operator" and next_type == "operator":
        return False
    if prev_type == "unknown" or next_type == "unknown":
        return True
    return True


def _prefers_alternative_with_geometry(
    char: str,
    alt: str,
    roi,
    context: str,
) -> bool:
    if roi is None:
        return False

    try:
        from segmentation.dual_head_utils import resolve_ambiguous_pair
    except ImportError:
        return False

    return resolve_ambiguous_pair(char, alt, roi, context) == alt


def _try_flip(char: str, conf: float, context: str, roi=None) -> Tuple[str, float]:
    if conf >= CORRECTION_CONF_THRESHOLD:
        return char, conf

    alternatives = ALTERNATIVES.get(char, [])
    if not alternatives:
        return char, conf

    char_type = _char_type(char)
    for alt in alternatives:
        alt_type = _char_type(alt)

        if context == "expected_digit" and char_type == "operator" and alt_type == "digit":
            if _prefers_alternative_with_geometry(char, alt, roi, context):
                print(f"[CTX] '{char}' -> '{alt}' (context={context}, conf={conf:.2f})")
                return alt, conf * 0.95
            print(f"[CTX] '{char}' -> '{alt}' (context={context}, conf={conf:.2f})")
            return alt, conf * 0.95

        if context == "expected_operator" and char_type == "digit" and alt_type == "operator":
            if _prefers_alternative_with_geometry(char, alt, roi, context):
                print(f"[CTX] '{char}' -> '{alt}' (context={context}, conf={conf:.2f})")
                return alt, conf * 0.95
            print(f"[CTX] '{char}' -> '{alt}' (context={context}, conf={conf:.2f})")
            return alt, conf * 0.95

    return char, conf


def _fix_double_operators(predictions: List[Dict[str, Any]], rois: Optional[List] = None) -> List[Dict[str, Any]]:
    result = list(predictions)
    for index in range(1, len(result)):
        if _char_type(result[index]["char"]) != "operator":
            continue
        if _char_type(result[index - 1]["char"]) != "operator":
            continue

        flip_index = index if result[index]["conf"] <= result[index - 1]["conf"] else index - 1
        item = result[flip_index]
        roi = rois[flip_index] if rois and flip_index < len(rois) else None
        new_char, new_conf = _try_flip(item["char"], item["conf"], "expected_digit", roi)
        if new_char != item["char"]:
            result[flip_index] = {
                **item,
                "char": new_char,
                "conf": new_conf,
                "corrected": True,
            }

    return result


def _fix_leading_operator(predictions: List[Dict[str, Any]], rois: Optional[List] = None) -> List[Dict[str, Any]]:
    if not predictions:
        return predictions

    result = list(predictions)
    if _char_type(result[0]["char"]) == "operator":
        item = result[0]
        roi = rois[0] if rois else None
        new_char, new_conf = _try_flip(item["char"], item["conf"], "expected_digit", roi)
        if new_char != item["char"]:
            result[0] = {**item, "char": new_char, "conf": new_conf, "corrected": True}

    return result


def _fix_trailing_operator(predictions: List[Dict[str, Any]], rois: Optional[List] = None) -> List[Dict[str, Any]]:
    if not predictions:
        return predictions

    result = list(predictions)
    if _char_type(result[-1]["char"]) == "operator" and result[-1]["conf"] < CORRECTION_CONF_THRESHOLD:
        item = result[-1]
        roi = rois[-1] if rois else None
        new_char, new_conf = _try_flip(item["char"], item["conf"], "expected_digit", roi)
        if new_char != item["char"]:
            result[-1] = {**item, "char": new_char, "conf": new_conf, "corrected": True}

    return result


def correct_sequence(
    predictions: List[Dict[str, Any]],
    roi_images: Optional[List] = None,
) -> List[Dict[str, Any]]:
    if not predictions:
        return predictions

    result = [{**prediction, "corrected": False} for prediction in predictions]
    result = _fix_leading_operator(result, roi_images)
    result = _fix_double_operators(result, roi_images)
    result = _fix_trailing_operator(result, roi_images)

    built: List[str] = []
    for index, item in enumerate(result):
        char = item["char"]
        conf = float(item["conf"])
        context = _expected_type(built)
        roi = roi_images[index] if roi_images and index < len(roi_images) else None

        prev_type = _char_type(built[-1]) if built else "start"
        char_type = _char_type(char)
        if not _is_valid_transition(prev_type, char_type) and conf < CORRECTION_CONF_THRESHOLD:
            new_char, new_conf = _try_flip(char, conf, context, roi)
            if new_char != char:
                result[index] = {**item, "char": new_char, "conf": new_conf, "corrected": True}
                char = new_char

        built.append(char)

    corrected_count = sum(1 for item in result if item.get("corrected"))
    if corrected_count:
        print(f"[CTX] Corrected {corrected_count}/{len(result)} characters using context")

    return result


def _resolve_predict_fn(predict_fn=None):
    if predict_fn is None:
        try:
            from model.operator_classifier import predict_character as _predict # operator_classifier moved to src/model
        except ImportError as exc:
            raise ImportError("Could not import model.operator_classifier.predict_character.") from exc
        predict_fn = _predict

    def _wrapped_predict_fn(roi, normalized=True):
        try: # dual_head_utils is still in src/segmentation
            return predict_fn(roi, normalized=normalized)
        except TypeError:
            return predict_fn(roi)

    return _wrapped_predict_fn


def _resolve_predict_top_k_fn(predict_top_k_fn=None):
    if predict_top_k_fn is None:
        try:
            from model.operator_classifier import predict_characters_top_k as _predict_top_k # operator_classifier moved to src/model
        except ImportError as exc:
            raise ImportError("Could not import model.operator_classifier.predict_characters_top_k.") from exc
        predict_top_k_fn = _predict_top_k

    def _wrapped_predict_top_k_fn(roi_images, normalized=True, top_k=5):
        try:
            return predict_top_k_fn(roi_images, normalized=normalized, top_k=top_k)
        except TypeError:
            return predict_top_k_fn(roi_images, top_k=top_k)

    return _wrapped_predict_top_k_fn


def _serialize_top_k_entries(entries) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for entry in entries or []:
        char = str((entry or {}).get("char") or "")
        conf = round(float((entry or {}).get("conf", 0.0)), 6)
        if not char:
            continue
        serialized.append({
            "char": char,
            "conf": conf,
        })
    return serialized


def build_raw_predictions(
    roi_images: list,
    predict_fn=None,
    use_dual_head: bool = True,
) -> List[Dict[str, Any]]:
    predict_fn = _resolve_predict_fn(predict_fn)

    if use_dual_head:
        try:
            from segmentation.dual_head_utils import predict_with_dual_head
        except ImportError:
            use_dual_head = False
        else:
            return [
                {"char": char, "conf": round(float(conf), 3)}
                for char, conf in (
                    predict_with_dual_head(roi, normalized=True, predict_fn=predict_fn)
                    for roi in roi_images
                )
            ]

    return [
        {"char": char, "conf": round(float(conf), 3)}
        for char, conf in (predict_fn(roi, normalized=True) for roi in roi_images)
    ]


def build_raw_predictions_with_top_k(
    roi_images: list,
    top_k: int = 5,
    predict_top_k_fn=None,
    use_dual_head: bool = True,
) -> List[Dict[str, Any]]:
    if not roi_images:
        return []

    predict_top_k_fn = _resolve_predict_top_k_fn(predict_top_k_fn)

    try:
        top_k_predictions = [
            _serialize_top_k_entries(item)
            for item in predict_top_k_fn(roi_images, normalized=True, top_k=top_k)
        ]
    except Exception:
        return [
            {
                **prediction,
                "top_k": [],
            }
            for prediction in build_raw_predictions(
                roi_images,
                use_dual_head=use_dual_head,
            )
        ]

    if len(top_k_predictions) < len(roi_images):
        top_k_predictions.extend([[] for _ in range(len(roi_images) - len(top_k_predictions))])
    else:
        top_k_predictions = top_k_predictions[:len(roi_images)]

    if use_dual_head:
        try:
            from segmentation.dual_head_utils import predict_with_dual_head
        except ImportError:
            use_dual_head = False

    results: List[Dict[str, Any]] = []
    for roi, entries in zip(roi_images, top_k_predictions):
        if not entries:
            fallback_predictions = build_raw_predictions([roi], use_dual_head=use_dual_head)
            fallback = fallback_predictions[0] if fallback_predictions else {"char": "", "conf": 0.0}
            results.append({
                **fallback,
                "top_k": [],
            })
            continue

        base_char = str(entries[0]["char"])
        base_conf = float(entries[0]["conf"])

        if use_dual_head:
            def _predict_prefetched(_roi, normalized=True, result=(base_char, base_conf)):
                return result

            char, conf = predict_with_dual_head(
                roi,
                normalized=True,
                predict_fn=_predict_prefetched,
            )
        else:
            char, conf = base_char, base_conf

        results.append({
            "char": char,
            "conf": round(float(conf), 3),
            "top_k": entries,
        })

    return results


def build_corrected_predictions(
    roi_images: list,
    predict_fn=None,
    use_dual_head: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sequence-level helper kept for compatibility.

    For multi-line inputs, prefer build_raw_predictions(...) first and apply
    correct_sequence(...) after grouping by line.
    """
    raw_predictions = build_raw_predictions(
        roi_images,
        predict_fn=predict_fn,
        use_dual_head=use_dual_head,
    )
    return correct_sequence(raw_predictions, roi_images)
