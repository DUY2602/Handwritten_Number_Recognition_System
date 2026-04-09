import cv2
import numpy as np

from .config import SEGMENTATION_CONFIG
from .logging_utils import get_logger, log_info_print

logger = get_logger(__name__)
print = lambda *args, **kwargs: log_info_print(*args, logger=logger, **kwargs)

def _to_ink_map(img):
    if img is None:
        raise ValueError("img must not be None")
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    edges = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if edges.mean() <= 35.0 and np.mean(gray >= 180) <= 0.45:
        return gray

    if len(img.shape) == 3:
        dark_from_gray  = cv2.subtract(255, gray)
        dark_from_color = cv2.subtract(255, img.min(axis=2).astype(np.uint8))
        return cv2.max(dark_from_gray, dark_from_color)

    return cv2.subtract(255, gray)

def _normalize_background(ink_map):
    """
    FIX-A: Loại bỏ gradient sáng/tối trải rộng (như Image 2, 3).

    Ý tưởng: ước tính background bằng cách blur rất mạnh (morphological CLOSE
    với kernel rất lớn), sau đó trừ nó ra. Kết quả: nét mực nổi đều dù background
    tối/sáng không đồng đều.

    Lưu ý: chỉ áp dụng khi std của ink_map cao (tức là có gradient mạnh).
    Nếu ảnh đã đều thì bỏ qua để tránh artifacts.
    """
    std = float(np.std(ink_map))
    if std < 18.0:
        # Ảnh đã đủ đều, không cần normalize background
        return ink_map

    # Kernel size = 1/6 chiều nhỏ hơn, tối thiểu 61px, phải lẻ
    h, w = ink_map.shape
    k = max(61, int(min(h, w) / 6) | 1)
    
    # [OPTIMIZATION] Morphological filter với kernel lớn (k > 100) trên ảnh phân giải cao
    # cực kỳ chậm. Ta sẽ thu nhỏ ảnh, filter trên ảnh nhỏ, rồi phóng to lại.
    max_dim = max(h, w)
    if max_dim > 600:
        scale = 600.0 / max_dim
        small_w, small_h = int(w * scale), int(h * scale)
        small_ink = cv2.resize(ink_map, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # Scale kernel tương ứng, tối thiểu 11px
        small_k = max(11, int(k * scale) | 1)
        
        bg_est_small = cv2.morphologyEx(
            small_ink,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_k, small_k))
        )
        bg_est = cv2.resize(bg_est_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg_est = cv2.morphologyEx(
            ink_map,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )

    # BUG FIX: phép divide loại gradient đúng cách, không invert
    # ink_map convention: CAO = mực, THẤP = nền
    # bg_est ≈ background (thấp ở vùng mực vì MORPH_CLOSE lấp đầy)
    # Sau divide: vùng mực (ink cao, bg thấp) → tỉ lệ cao → nét mực nổi đều
    bg_f   = np.clip(bg_est.astype(np.float32), 1.0, 255.0)
    ink_f  = ink_map.astype(np.float32)
    normalized = np.clip((ink_f / bg_f) * 128.0, 0, 255).astype(np.uint8)

    # Sanity check: nếu normalize lại bị tối (mean < 5) → bg_est quá thấp → fallback
    if float(np.mean(normalized)) < 5.0:
        print("[BG-NORM] fallback to original ink_map (normalized image too dark)")
        return ink_map

    print(
        f"[BG-NORM] std={std:.1f}, k={k}, "
        f"mean: {float(np.mean(ink_map)):.1f} -> {float(np.mean(normalized)):.1f}"
    )
    return normalized

def _enhance(ink_map):
    """
    FIX-4: CLAHE clipLimit thích nghi theo độ sáng.
    FIX-A: Thêm _normalize_background() trước CLAHE.
    """
    # FIX-A: normalize trước
    ink_map = _normalize_background(ink_map)

    mean_brightness = float(np.mean(ink_map))
    clip = 4.0 if mean_brightness < 30 else 3.0 if mean_brightness < 60 else 2.0
    clahe     = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    equalized = clahe.apply(ink_map)

    blurred   = cv2.GaussianBlur(equalized, (5, 5), 0)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat    = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, bg_kernel)

    return cv2.add(blurred, tophat)

def _score_threshold(binary):
    cfg = SEGMENTATION_CONFIG.threshold_score
    fg = int(np.count_nonzero(binary))
    if fg == 0:
        return float("-inf")

    ratio      = fg / float(binary.size)
    edges      = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    edge_ratio = np.count_nonzero(edges) / max(1.0, float(edges.size))

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small_limit = max(6, int(binary.size * 0.00004))
    noise_count = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] < small_limit)) if n_labels > 1 else 0
    if n_labels > 1:
        widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
        heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        lineish_count = int(np.count_nonzero(
            (widths >= binary.shape[1] * cfg.lineish_min_width_ratio)
            & (heights <= max(cfg.lineish_max_height_abs, int(binary.shape[0] * cfg.lineish_max_height_ratio)))
        ))
        charish_count = int(np.count_nonzero(
            (heights >= max(cfg.charish_min_height_abs, int(binary.shape[0] * cfg.charish_min_height_ratio)))
            & (widths <= max(cfg.charish_max_width_abs, int(binary.shape[1] * cfg.charish_max_width_ratio)))
        ))
    else:
        lineish_count = 0
        charish_count = 0

    return (
        -abs(ratio - cfg.target_fg_ratio)
        - (edge_ratio * cfg.edge_ratio_weight)
        - (min(noise_count, cfg.noise_count_cap) * cfg.noise_weight)
        - (lineish_count * cfg.lineish_weight)
        + (min(charish_count, cfg.charish_cap) * cfg.charish_weight)
    )

def _remove_small_cc(binary, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    kept = False
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 255
            kept = True
    if not kept and n > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out[labels == best] = 255
    return out

def _binarize(enhanced, h_img, w_img):
    """FIX-2: Cap blockSize ở 51 + candidate C=-5."""
    min_cc = max(6, int(h_img * w_img * 0.00002))

    raw_block  = max(21, int(round(min(h_img, w_img) * 0.04)) | 1)
    block_size = min(raw_block, 51)

    def _adaptive(c_val):
        return cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c_val,
        )

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates_raw = [_adaptive(-2), _adaptive(-5), otsu]

    def prepare(b):
        b = cv2.medianBlur(b, 3)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return _remove_small_cc(b, min_cc)

    candidates = [prepare(b) for b in candidates_raw]
    return max(candidates, key=_score_threshold)

def _binarize_dark_strokes(gray, h_img, w_img):
    """
    Extract dark pen strokes on bright paper with a blackhat transform.

    This is more robust than adaptive thresholding alone on phone photos with
    paper texture, shadows, or gradual lighting changes.
    """
    raw_kernel = max(15, int(round(min(h_img, w_img) * 0.05)) | 1)
    kernel_size = min(raw_kernel, 61)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.medianBlur(binary, 3)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    min_cc = max(20, int(h_img * w_img * 0.00003))
    return _remove_small_cc(binary, min_cc)

def _extract_horizontal_line_mask(
    binary,
    min_span_ratio=SEGMENTATION_CONFIG.horizontal_line_mask.min_span_ratio,
    bridge_gap_ratio=SEGMENTATION_CONFIG.horizontal_line_mask.bridge_gap_ratio,
    max_line_height_ratio=SEGMENTATION_CONFIG.horizontal_line_mask.max_line_height_ratio,
    lower_bias=False,
):
    if binary is None or getattr(binary, "size", 0) == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    h_img, w_img = binary.shape[:2]
    if h_img < 4 or w_img < 16:
        return np.zeros_like(binary)

    min_span = max(24, int(round(w_img * min_span_ratio)))
    bridge_gap = max(5, int(round(w_img * bridge_gap_ratio)))

    healed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_gap, 1)),
        iterations=1,
    )
    seed = cv2.morphologyEx(
        healed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (min_span, 1)),
        iterations=1,
    )
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, bridge_gap // 2), 1)),
        iterations=1,
    )

    if np.count_nonzero(seed) == 0:
        return np.zeros_like(binary)

    max_height = max(4, int(round(h_img * max_line_height_ratio)))
    mask = np.zeros_like(binary)
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_span:
            continue
        if lower_bias and (y + h / 2.0) < h_img * 0.40:
            continue

        component = seed[y:y + h, x:x + w]
        if h > max_height:
            row_counts = np.count_nonzero(component, axis=1).astype(np.float32)
            if row_counts.size == 0 or float(np.max(row_counts)) < max(8.0, w * 0.45):
                continue
            peak = int(np.argmax(row_counts))
            half = max(1, min(max_height // 2, 3))
            ry0 = max(0, peak - half)
            ry1 = min(h, peak + half + 1)
            component = component[ry0:ry1, :]
            y += ry0
            h = component.shape[0]

        mask[y:y + h, x:x + w] = cv2.max(mask[y:y + h, x:x + w], component)

    return mask

def _extract_vertical_guide_mask(
    binary,
    min_height_ratio=0.72,
    max_width_ratio=0.030,
    repeated_count=4,
):
    if binary is None or getattr(binary, "size", 0) == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    h_img, w_img = binary.shape[:2]
    if h_img < 8 or w_img < 8:
        return np.zeros_like(binary)

    kernel_h = max(24, int(round(h_img * min_height_ratio)))
    seed = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h)),
        iterations=1,
    )
    if np.count_nonzero(seed) == 0:
        return np.zeros_like(binary)

    candidates = []
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < max(24, int(h_img * min_height_ratio)):
            continue
        if w > max(8, int(round(w_img * max_width_ratio))):
            continue
        roi = seed[y:y + h, x:x + w]
        fill_ratio = np.count_nonzero(roi) / max(1.0, float(roi.size))
        if fill_ratio < 0.05 or fill_ratio > 0.92:
            continue
        candidates.append((x, y, w, h))

    if not candidates:
        return np.zeros_like(binary)

    if len(candidates) < repeated_count:
        candidates = [
            rect
            for rect in candidates
            if rect[0] <= 2 or (rect[0] + rect[2]) >= (w_img - 2)
        ]
        if not candidates:
            return np.zeros_like(binary)

    mask = np.zeros_like(binary)
    for x, y, w, h in candidates:
        mask[y:y + h, x:x + w] = cv2.max(mask[y:y + h, x:x + w], seed[y:y + h, x:x + w])
    return mask

def _remove_grid_lines(binary, h_img, w_img):
    """FIX-1: Tách heal ra ngoài subtract, iterations=3."""
    h_len = max(40, int(w_img * 0.25))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    h_lines = cv2.max(
        h_lines,
        _extract_horizontal_line_mask(
            binary,
            min_span_ratio=0.28,
            bridge_gap_ratio=0.05,
            max_line_height_ratio=0.10,
            lower_bias=False,
        ),
    )
    v_lines = _extract_vertical_guide_mask(
        binary,
        min_height_ratio=0.78,
        max_width_ratio=0.022,
        repeated_count=4,
    )

    lines = cv2.add(h_lines, v_lines)
    cleaned = cv2.subtract(binary, lines)
    cleaned = cv2.medianBlur(cleaned, 3)

    n_removed = int(np.count_nonzero(lines))
    print(f"[LINE] Da xoa ~{n_removed} pixel duong ke bang/dong.")
    return cleaned

def _binarize_canvas_strokes(gray, h_img, w_img):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    thresh = _remove_small_cc(thresh, max(6, int(h_img * w_img * 0.00002)))
    return thresh
