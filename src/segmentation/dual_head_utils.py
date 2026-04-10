"""
dual_head_utils.py
==================
Aspect-ratio pre-filter and dual-head scoring for digit + operator classification.

Position in pipeline (directly replaces predict_character):
    from dual_head_utils import predict_with_dual_head
    char, conf = predict_with_dual_head(roi, normalized=True)

Requirements:
    - operator_classifier.predict_character  (already available)
    - MNIST / digit model via operator_classifier (already available)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# ─── Configuration constants ────────────────────────────────────────────────────────

# Aspect ratio w/h of ROI after normalization to tight bounding box
_ASPECT_DIGIT_MIN   = 0.20   # digit "1" is very narrow
_ASPECT_DIGIT_MAX   = 1.80   # digit "0" is slightly wider than high
_ASPECT_OP_MINUS    = (2.5, 9.0)   # "-" : very wide
_ASPECT_OP_EQUAL    = (1.8, 6.0)   # "=" : wide, consisting of 2 strokes
_ASPECT_OP_PLUS     = (0.6, 1.6)   # "+" : near square
_ASPECT_OP_MUL      = (0.5, 1.8)   # "×" : near square
_ASPECT_OP_DIV      = (0.3, 1.0)   # "÷" : higher than wide

# Minimum confidence threshold to accept prediction
CONF_THRESHOLD = 0.45

# Commonly confused characters and aspect ratio for distinction
# Format: (char_a, char_b, aspect_threshold, "a_if_below" / "b_if_below")
_AMBIGUOUS_ASPECT_RULES = [
    # "1" (narrow) vs "-" (horizontal) — if h > w×1.5 then it's "1"
    ("1", "-",  1.0, "below_is_digit"),
    # "7" (slanted) vs "/" (slanted) — "/" usually taller/wider than "7"
    ("7", "/",  0.8, "below_is_op"),
]

# Operator symbols including both Unicode and ASCII forms
OPERATOR_CHARS = {"+", "-", "×", "÷", "/", "=", "*", "x", "X"}
DIGIT_CHARS    = set("0123456789")


# ─── Helper: tight aspect ratio từ binary image ──────────────────────────────

def _tight_aspect(roi: np.ndarray) -> float:
    """Calculate w/h of tight bounding box (removes white padding)."""
    binary = np.where(roi > 0, 255, 0).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 1.0
    _, _, w, h = cv2.boundingRect(coords)
    return float(w) / max(1.0, float(h))


def _pixel_symmetry(roi: np.ndarray) -> float:
    """
    Measure horizontal symmetry: '-' and '=' have strokes concentrated at middle height,
    while '1' and '/' have strokes spread from top to bottom.
    Return pixel ratio in the middle 40% / total pixels (0→1).
    """
    if roi.size == 0 or np.count_nonzero(roi) == 0:
        return 0.5
    h = roi.shape[0]
    top     = max(0, int(h * 0.30))
    bottom  = min(h, int(h * 0.70))
    center_pixels = int(np.count_nonzero(roi[top:bottom]))
    total_pixels  = int(np.count_nonzero(roi))
    return float(center_pixels) / max(1, total_pixels)


def _stroke_density_split(roi: np.ndarray) -> float:
    """
    Ratio between "active" rows and total rows.
    '=' and '-' have few active rows (horizontal stroke), '1' has many active rows.
    """
    if roi.size == 0:
        return 0.5
    active_rows = np.count_nonzero(np.count_nonzero(roi, axis=1))
    return float(active_rows) / max(1, roi.shape[0])


# ─── Rule-based aspect pre-filter ────────────────────────────────────────────

def aspect_prefilter(roi: np.ndarray) -> str | None:
    """
    Based on pure shape, eliminate obvious false possibilities.

    Returns:
        "digit"    — certainly a digit
        "operator" — certainly an operator
        None       — not sure, need model
    """
    aspect = _tight_aspect(roi)
    symmetry = _pixel_symmetry(roi)
    density  = _stroke_density_split(roi)

    # Clearly horizontal (w >> h): can only be "-" or "="
    if aspect >= _ASPECT_OP_MINUS[0]:
        return "operator"

    # Near square with high symmetry (many pixels in middle): could be "+" or "×"
    if 0.65 <= aspect <= 1.55 and symmetry >= 0.55 and density >= 0.55:
        return "operator"

    # Narrow and tall: "1", "/", "÷"
    if aspect < 0.35 and density >= 0.65:
        # Cannot conclude immediately — need model
        return None

    # Aspect ratio within normal digit range
    if _ASPECT_DIGIT_MIN <= aspect <= _ASPECT_DIGIT_MAX:
        return None  # Need model confirmation

    return None


# ─── Ambiguous pair resolution ───────────────────────────────────────────────

def resolve_ambiguous_pair(
    char: str,
    alt_char: str,
    roi: np.ndarray,
    context: str,
) -> str:
    """
    When model returns char with low confidence and char/alt_char are commonly confused pairs,
    use geometric features to select the correct one.

    Args:
        char     : predicted character from model
        alt_char : alternative character (confused pair)
        roi      : 28×28 binary image
        context  : "expected_digit" | "expected_operator"

    Returns:
        Resolved character.
    """
    aspect   = _tight_aspect(roi)
    symmetry = _pixel_symmetry(roi)
    density  = _stroke_density_split(roi)

    pair = frozenset([char, alt_char])

    # Pair "1" vs "-"
    if pair == frozenset(["1", "-"]):
        # "-" horizontal: high aspect, high symmetry, low density
        # "1" vertical: low aspect, high density
        if aspect >= 2.0 and symmetry >= 0.60:
            return "-"
        if aspect < 0.8 and density >= 0.60:
            return "1"
        # Use context as tiebreaker
        return "1" if context == "expected_digit" else "-"

    # Pair "7" vs "/"
    if pair == frozenset(["7", "/"]):
        # "/" has no top horizontal stroke, "7" does
        top_density = float(np.count_nonzero(roi[:roi.shape[0] // 4])) / max(1, roi.shape[0] // 4 * roi.shape[1])
        if top_density >= 0.10:
            return "7"   # Has top horizontal stroke = "7"
        return "/" if context == "expected_operator" else "7"

    # Pair "×" vs "x" / "X"
    if pair in (frozenset(["×", "x"]), frozenset(["×", "X"]), frozenset(["x", "X"])):
        return "×" if context == "expected_operator" else char

    # Pair "0" vs "O" (if model accepts alphanumeric)
    if pair == frozenset(["0", "O"]):
        return "0" if context == "expected_digit" else char

    # Pair "+" vs "t"
    if pair == frozenset(["+", "t"]):
        # "+" has horizontal stroke in middle; "t" has horizontal stroke at top 1/3
        center_y = int(roi.shape[0] * 0.45)
        top_y    = int(roi.shape[0] * 0.25)
        center_row_density = float(np.count_nonzero(roi[center_y - 2:center_y + 2]))
        top_row_density    = float(np.count_nonzero(roi[top_y - 2:top_y + 2]))
        if center_row_density >= top_row_density:
            return "+"
        return "t" if context == "expected_digit" else "+"

    return char


# ─── Main API ────────────────────────────────────────────────────────────────

def predict_with_dual_head(
    roi: np.ndarray,
    normalized: bool = True,
    predict_fn=None, # operator_classifier moved to src/model
) -> Tuple[str, float]:
    """
    Wrapper around original predict_character, adding:
    1. Aspect-ratio pre-filter (eliminate unreasonable predictions based on shape)
    2. Geometric tiebreaker for commonly confused pairs

    Args:
        roi         : normalized 28×28 image (white-on-black)
        normalized  : True if roi is already standard 28x28 binary
        predict_fn  : callable(roi, normalized) → (char, conf)
                      If None, import operator_classifier.predict_character automatically

    Returns:
        Improved (char, conf)
    """
    if predict_fn is None:
        try:
            from model.operator_classifier import predict_character as _default_predict # operator_classifier moved to src/model
            predict_fn = _default_predict
        except ImportError:
            raise ImportError(
                "model.operator_classifier not found. "
                "Pass predict_fn or ensure PYTHONPATH is correct."
            )

    char, conf = predict_fn(roi, normalized=normalized)

    # ── 1. Aspect hint: if prefilter shows mismatch, reduce confidence ──
    hint = aspect_prefilter(roi)
    if hint == "digit" and char in OPERATOR_CHARS:
        conf *= 0.60
    elif hint == "operator" and char in DIGIT_CHARS:
        conf *= 0.60

    # ── 2. With low confidence, try resolving ambiguous pairs ──────────────────
    if conf < 0.75:
        aspect = _tight_aspect(roi)

        # Determine context hint từ aspect
        if aspect >= 1.8:
            geo_context = "expected_operator"
        elif aspect <= 0.5:
            geo_context = "ambiguous"
        else:
            geo_context = "ambiguous"

        AMBIGUOUS_PAIRS = [
            ("1", "-"), ("-", "1"),
            ("7", "/"), ("/", "7"),
            ("×", "x"), ("x", "×"),
            ("×", "X"), ("X", "×"),
            ("0", "O"), ("O", "0"),
            ("+", "t"), ("t", "+"),
        ]
        for a, b in AMBIGUOUS_PAIRS:
            if char == a:
                resolved = resolve_ambiguous_pair(a, b, roi, geo_context)
                if resolved != char:
                    print(f"[DUAL] Ambiguous '{char}'->'{resolved}' (conf={conf:.2f}, aspect={aspect:.2f})")
                    char = resolved
                break

    return char, conf


# ─── Batch helper ─────────────────────────────────────────────────────────────

def predict_sequence_dual_head(
    roi_images: list,
    normalized: bool = True,
    predict_fn=None,
) -> list:
    """
    Predict entire ROI sequence, return list of {"char": str, "conf": float}.
    Substitute for predict_character loop in app.py.
    """
    results = []
    for roi in roi_images:
        char, conf = predict_with_dual_head(roi, normalized=normalized, predict_fn=predict_fn)
        results.append({"char": char, "conf": round(float(conf), 3)})
    return results
