"""
dual_head_utils.py
==================
Aspect-ratio pre-filter và dual-head scoring cho digit + operator classification.

Vị trí trong pipeline (thay thế predict_character trực tiếp):
    from dual_head_utils import predict_with_dual_head
    char, conf = predict_with_dual_head(roi, normalized=True)

Yêu cầu:
    - operator_classifier.predict_character  (đã có sẵn)
    - MNIST / digit model thông qua operator_classifier (đã có sẵn)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# ─── Hằng số cấu hình ────────────────────────────────────────────────────────

# Aspect ratio w/h của ROI sau khi normalize về tight bounding box
_ASPECT_DIGIT_MIN   = 0.20   # số "1" rất hẹp
_ASPECT_DIGIT_MAX   = 1.80   # số "0" hơi rộng hơn cao
_ASPECT_OP_MINUS    = (2.5, 9.0)   # "-" : rất rộng
_ASPECT_OP_EQUAL    = (1.8, 6.0)   # "=" : rộng, gồm 2 nét
_ASPECT_OP_PLUS     = (0.6, 1.6)   # "+" : gần vuông
_ASPECT_OP_MUL      = (0.5, 1.8)   # "×" : gần vuông
_ASPECT_OP_DIV      = (0.3, 1.0)   # "÷" : cao hơn rộng

# Ngưỡng confidence tối thiểu để chấp nhận prediction
CONF_THRESHOLD = 0.45

# Ký tự hay bị nhầm và aspect ratio để phân biệt
# Format: (char_a, char_b, aspect_threshold, "a_if_below" / "b_if_below")
_AMBIGUOUS_ASPECT_RULES = [
    # "1" (hẹp) vs "-" (ngang) — nếu h > w×1.5 thì là "1"
    ("1", "-",  1.0, "below_is_digit"),
    # "7" (nghiêng) vs "/" (nghiêng) — "/" thường cao hơn rộng hơn "7"
    ("7", "/",  0.8, "below_is_op"),
]

# Operator symbols bao gồm cả dạng Unicode và ASCII
OPERATOR_CHARS = {"+", "-", "×", "÷", "/", "=", "*", "x", "X"}
DIGIT_CHARS    = set("0123456789")


# ─── Helper: tight aspect ratio từ binary image ──────────────────────────────

def _tight_aspect(roi: np.ndarray) -> float:
    """Tính w/h của tight bounding box (loại bỏ padding trắng)."""
    binary = np.where(roi > 0, 255, 0).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 1.0
    _, _, w, h = cv2.boundingRect(coords)
    return float(w) / max(1.0, float(h))


def _pixel_symmetry(roi: np.ndarray) -> float:
    """
    Đo tính đối xứng ngang: ký tự '-' và '=' có nét tập trung ở giữa chiều cao,
    trong khi '1' và '/' có nét trải đều từ trên xuống dưới.
    Trả về tỉ lệ pixel trong 40% giữa / tổng pixel (0→1).
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
    Tỉ lệ giữa số hàng "active" và tổng hàng.
    '=' và '-' có ít hàng active (nét nằm ngang), '1' có nhiều hàng active.
    """
    if roi.size == 0:
        return 0.5
    active_rows = np.count_nonzero(np.count_nonzero(roi, axis=1))
    return float(active_rows) / max(1, roi.shape[0])


# ─── Rule-based aspect pre-filter ────────────────────────────────────────────

def aspect_prefilter(roi: np.ndarray) -> str | None:
    """
    Dựa trên hình dạng thuần túy, loại bỏ các khả năng hiển nhiên sai.

    Returns:
        "digit"    — chắc chắn là chữ số
        "operator" — chắc chắn là toán tử
        None       — không chắc, cần dùng model
    """
    aspect = _tight_aspect(roi)
    symmetry = _pixel_symmetry(roi)
    density  = _stroke_density_split(roi)

    # Nằm ngang rõ ràng (w >> h): chỉ có thể là "-" hoặc "="
    if aspect >= _ASPECT_OP_MINUS[0]:
        return "operator"

    # Gần vuông với symmetry cao (nhiều pixel ở giữa): có thể là "+" hoặc "×"
    if 0.65 <= aspect <= 1.55 and symmetry >= 0.55 and density >= 0.55:
        return "operator"

    # Hẹp và cao: "1", "/", "÷"
    if aspect < 0.35 and density >= 0.65:
        # Không thể kết luận ngay — cần model
        return None

    # Aspect ratio trong khoảng bình thường của digit
    if _ASPECT_DIGIT_MIN <= aspect <= _ASPECT_DIGIT_MAX:
        return None  # Cần model xác nhận

    return None


# ─── Ambiguous pair resolution ───────────────────────────────────────────────

def resolve_ambiguous_pair(
    char: str,
    alt_char: str,
    roi: np.ndarray,
    context: str,
) -> str:
    """
    Khi model trả về char với confidence thấp và char/alt_char là cặp hay nhầm,
    dùng geometric features để chọn cái đúng hơn.

    Args:
        char     : ký tự model predict
        alt_char : ký tự alternative (cặp hay nhầm)
        roi      : ảnh 28×28 binary
        context  : "expected_digit" | "expected_operator"

    Returns:
        Ký tự đã được resolve.
    """
    aspect   = _tight_aspect(roi)
    symmetry = _pixel_symmetry(roi)
    density  = _stroke_density_split(roi)

    pair = frozenset([char, alt_char])

    # Cặp "1" vs "-"
    if pair == frozenset(["1", "-"]):
        # "-" nằm ngang: aspect cao, symmetry cao, density thấp
        # "1" thẳng đứng: aspect thấp, density cao
        if aspect >= 2.0 and symmetry >= 0.60:
            return "-"
        if aspect < 0.8 and density >= 0.60:
            return "1"
        # Dùng context làm tiebreaker
        return "1" if context == "expected_digit" else "-"

    # Cặp "7" vs "/"
    if pair == frozenset(["7", "/"]):
        # "/" không có nét ngang ở trên, "7" có
        top_density = float(np.count_nonzero(roi[:roi.shape[0] // 4])) / max(1, roi.shape[0] // 4 * roi.shape[1])
        if top_density >= 0.10:
            return "7"   # Có nét ngang trên = "7"
        return "/" if context == "expected_operator" else "7"

    # Cặp "×" vs "x" / "X"
    if pair in (frozenset(["×", "x"]), frozenset(["×", "X"]), frozenset(["x", "X"])):
        return "×" if context == "expected_operator" else char

    # Cặp "0" vs "O" (nếu model nhận cả alphanumeric)
    if pair == frozenset(["0", "O"]):
        return "0" if context == "expected_digit" else char

    # Cặp "+" vs "t"
    if pair == frozenset(["+", "t"]):
        # "+" có nét ngang ngay giữa; "t" có nét ngang ở khoảng 1/3 trên
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
    predict_fn=None,
) -> Tuple[str, float]:
    """
    Wrapper quanh predict_character gốc, thêm:
    1. Aspect-ratio pre-filter (loại bỏ dự đoán không hợp lý theo hình dạng)
    2. Geometric tiebreaker cho các cặp hay nhầm

    Args:
        roi         : ảnh 28×28 đã normalize (white-on-black)
        normalized  : True nếu roi đã là 28×28 binary chuẩn
        predict_fn  : callable(roi, normalized) → (char, conf)
                      Nếu None, import operator_classifier.predict_character tự động

    Returns:
        (char, conf) đã được cải thiện
    """
    if predict_fn is None:
        try:
            from segmentation.operator_classifier import predict_character as _default_predict
            predict_fn = _default_predict
        except ImportError:
            raise ImportError(
                "Không tìm thấy operator_classifier. "
                "Truyền predict_fn vào hoặc đảm bảo PYTHONPATH đúng."
            )

    char, conf = predict_fn(roi, normalized=normalized)

    # ── 1. Aspect hint: nếu prefilter cho thấy không khớp, giảm confidence ──
    hint = aspect_prefilter(roi)
    if hint == "digit" and char in OPERATOR_CHARS:
        conf *= 0.60
    elif hint == "operator" and char in DIGIT_CHARS:
        conf *= 0.60

    # ── 2. Với confidence thấp, thử resolve ambiguous pairs ──────────────────
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
    Predict toàn bộ sequence ROI, trả về list {"char": str, "conf": float}.
    Dùng thay thế vòng lặp predict_character trong app.py.
    """
    results = []
    for roi in roi_images:
        char, conf = predict_with_dual_head(roi, normalized=normalized, predict_fn=predict_fn)
        results.append({"char": char, "conf": round(float(conf), 3)})
    return results
