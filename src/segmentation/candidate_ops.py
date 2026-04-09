import cv2
import numpy as np

from .config import SEGMENTATION_CONFIG
from .image_ops import _extract_horizontal_line_mask, _extract_vertical_guide_mask, _remove_small_cc
from .logging_utils import get_logger, log_info_print
from .rect_ops import (
    _collect_component_rects,
    _count_connected_components_in_roi,
    _dedupe_regions,
    _extract_rects_from_thresh,
    _filter_rects,
    _is_line_dominated,
    _is_line_like_rect,
    _merge_broken_parts,
    _merge_overlapping_regions,
    _merge_single_digit_pairs,
    _operator_preserving_filter,
    _rect_fill_ratio,
    _rect_overlap_ratio,
    _remove_inside_boxes,
    _sort_by_row_col,
    _split_wide_rects,
)

logger = get_logger(__name__)
print = lambda *args, **kwargs: log_info_print(*args, logger=logger, **kwargs)

def _build_colored_ink_mask(img, gray, h_img, w_img):
    """Highlight colored pen strokes while suppressing printed black content."""
    if img is None or len(img.shape) != 3 or img.shape[2] < 3:
        return None

    cfg = SEGMENTATION_CONFIG.colored_ink_mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.uint8)
    spread = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    darkness = cv2.subtract(255, gray)

    candidate_zone = darkness >= max(18, int(np.percentile(darkness, 55) * 0.35))
    if not np.any(candidate_zone):
        return None

    sat_samples = sat[candidate_zone]
    spread_samples = spread[candidate_zone]
    sat_thresh = int(np.clip(
        np.percentile(sat_samples, cfg.sat_percentile),
        cfg.sat_threshold_min,
        cfg.sat_threshold_max,
    ))
    spread_thresh = int(np.clip(
        np.percentile(spread_samples, cfg.sat_percentile),
        cfg.spread_threshold_min,
        cfg.spread_threshold_max,
    ))
    dark_thresh = max(26, int(np.percentile(darkness[candidate_zone], 40)))

    strong_sat = sat >= sat_thresh
    strong_spread = spread >= spread_thresh
    dark_enough = darkness >= dark_thresh

    relaxed_sat = max(cfg.relaxed_sat_floor, int(round(sat_thresh * cfg.relaxed_sat_ratio)))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[
        dark_enough
        & (
            strong_sat
            | (strong_spread & (sat >= relaxed_sat))
        )
    ] = 255

    if np.count_nonzero(mask) == 0:
        return None

    close_k = max(5, int(round(min(h_img, w_img) * 0.0025)) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        iterations=1,
    )
    min_cc = max(24, int(h_img * w_img * 0.00001))
    return _remove_small_cc(mask, min_cc)

def _select_colored_focus_row(mask, w_img, h_img):
    if mask is None or np.count_nonzero(mask) == 0:
        return None

    cfg = SEGMENTATION_CONFIG.focus_row
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(40, int(h_img * w_img * 0.000012))
    components = []

    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        if area < min_area or h < max(cfg.min_height_abs, int(h_img * cfg.min_height_ratio)) or w < 4:
            continue
        if w >= w_img * 0.45 and h <= max(18, int(h_img * 0.03)):
            continue

        components.append({
            "label": lbl,
            "rect": (x, y, w, h),
            "area": area,
        })

    if len(components) < 4:
        return None

    med_h = float(np.median([item["rect"][3] for item in components]))
    row_snap = max(cfg.row_snap_min, int(med_h * cfg.row_snap_ratio))

    centers = sorted(
        ((item["rect"][1] + item["rect"][3] / 2.0), idx)
        for idx, item in enumerate(components)
    )
    groups = []
    current = [centers[0][1]]
    prev_cy = centers[0][0]
    for cy, idx in centers[1:]:
        if cy - prev_cy > row_snap:
            groups.append([components[item_idx] for item_idx in current])
            current = []
        current.append(idx)
        prev_cy = cy
    if current:
        groups.append([components[item_idx] for item_idx in current])

    def _score_group(group):
        rects = [item["rect"] for item in group]
        xs = [rect[0] for rect in rects]
        ys = [rect[1] for rect in rects]
        x2s = [rect[0] + rect[2] for rect in rects]
        y2s = [rect[1] + rect[3] for rect in rects]

        span_ratio = (max(x2s) - min(xs)) / max(1.0, float(w_img))
        if len(group) < cfg.min_group_size or span_ratio < cfg.min_span_ratio:
            return float("-inf")

        heights = [rect[3] for rect in rects]
        widths = [rect[2] for rect in rects]
        med_h_group = float(np.median(heights))
        med_w_group = float(np.median(widths))
        med_a_group = float(np.median([item["area"] for item in group]))
        region_area = max(1.0, float((max(x2s) - min(xs)) * (max(y2s) - min(ys))))
        coverage = sum(item["area"] for item in group) / region_area
        line_like = sum(1 for rect in rects if _is_line_like_rect(rect, w_img, h_img))
        giant = sum(
            1
            for rect in rects
            if rect[2] * rect[3] >= med_a_group * 5.0
        )
        row_height_ratio = (max(y2s) - min(ys)) / max(1.0, med_h_group)

        score = 0.0
        score += len(group) * 2.4
        score += min(8.0, span_ratio * 10.0)
        score += min(3.0, coverage * 6.0)
        score += min(2.5, med_h_group / max(20.0, h_img * 0.035))
        score -= line_like * 4.0
        score -= giant * 1.8
        score -= max(0.0, row_height_ratio - 2.8) * 1.2
        if med_w_group <= max(6.0, med_h_group * 0.14):
            score -= 1.0
        return score

    best_group = None
    best_score = float("-inf")
    for group in groups:
        score = _score_group(group)
        if score > best_score:
            best_score = score
            best_group = group

    if best_group is None or best_score < cfg.min_best_score:
        return None

    rects = [item["rect"] for item in best_group]
    xs = [rect[0] for rect in rects]
    ys = [rect[1] for rect in rects]
    x2s = [rect[0] + rect[2] for rect in rects]
    y2s = [rect[1] + rect[3] for rect in rects]
    med_w = float(np.median([rect[2] for rect in rects]))
    med_h_group = float(np.median([rect[3] for rect in rects]))

    focus_mask = np.zeros_like(mask)
    for item in best_group:
        focus_mask[labels == item["label"]] = 255

    dilate_k = max(15, int(round(min(h_img, w_img) * 0.012)) | 1)
    focus_mask = cv2.dilate(
        focus_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)),
        iterations=1,
    )

    pad_x = max(40, int(med_w * 1.05))
    pad_y = max(30, int(med_h_group * 0.9))
    region = (
        max(0, int(min(xs) - pad_x)),
        max(0, int(min(ys) - pad_y)),
        min(w_img, int(max(x2s) + pad_x)),
        min(h_img, int(max(y2s) + pad_y)),
    )

    return {
        "score": best_score,
        "rects": rects,
        "mask": focus_mask,
        "region": region,
    }

def _segment_guided_handwriting_row(img, gray, thresh, w_img, h_img):
    color_mask = _build_colored_ink_mask(img, gray, h_img, w_img)
    focus_row = _select_colored_focus_row(color_mask, w_img, h_img)
    if focus_row is None:
        return None

    candidate_region = _tighten_region_to_row_band(thresh, focus_row["region"])

    def _score_candidate(rects, source_thresh):
        is_valid, reason = _validate_generic_result(rects, source_thresh, w_img, h_img)
        if not is_valid:
            return float("-inf"), reason
        if not rects:
            return float("-inf"), "No rects"

        widths = [r[2] for r in rects]
        heights = [r[3] for r in rects]
        areas = [r[2] * r[3] for r in rects]
        med_w = float(np.median(widths))
        med_h = float(np.median(heights))
        med_a = float(np.median(areas))
        ratios = np.asarray(widths, dtype=np.float32) / np.maximum(1.0, np.asarray(heights, dtype=np.float32))
        width_cv = float(np.std(widths) / max(1.0, med_w))
        tiny_count = sum(
            1
            for x, y, w, h in rects
            if (
                w <= max(18.0, med_w * 0.38)
                or h <= max(26.0, med_h * 0.42)
                or (w * h) <= max(180.0, med_a * 0.24)
            )
        )
        line_like = sum(1 for rect in rects if _is_line_like_rect(rect, w_img, h_img))
        min_x = min(rect[0] for rect in rects)
        max_x = max(rect[0] + rect[2] for rect in rects)
        span_ratio = (max_x - min_x) / max(1.0, float(w_img))
        fill_values = [_rect_fill_ratio(rect, source_thresh) for rect in rects]
        median_fill = float(np.median(fill_values)) if fill_values else 0.0
        slim_ratio = float(np.mean(ratios < 0.18)) if ratios.size else 0.0
        tiny_ratio = tiny_count / max(1.0, float(len(rects)))
        wide_count = sum(1 for width in widths if width >= max(110.0, med_w * 1.60))

        score = 0.0
        score += min(40.0, len(rects) * 4.0)
        score += min(3.0, span_ratio * 4.0)
        score += min(2.0, median_fill * 5.0)
        score -= max(0.0, len(rects) - 12) * 2.0
        score -= tiny_count * 2.2
        score -= line_like * 4.0
        score -= max(0.0, slim_ratio - 0.22) * 18.0
        score -= tiny_ratio * 10.0
        score -= max(0.0, width_cv - 0.75) * 6.0
        score -= wide_count * 3.0
        return score, reason

    candidates = []

    region_rects = _segment_generic_region(thresh, candidate_region, w_img, h_img)
    region_score, region_reason = _score_candidate(region_rects, thresh)
    if np.isfinite(region_score):
        candidates.append({
            "mode": "region",
            "score": region_score,
            "reason": region_reason,
            "rects": region_rects,
            "thresh": thresh,
        })

    guided_thresh = cv2.bitwise_and(thresh, focus_row["mask"])
    masked_region = _tighten_region_to_row_band(guided_thresh, candidate_region)
    masked_rects = _segment_generic_region(guided_thresh, masked_region, w_img, h_img)
    masked_score, masked_reason = _score_candidate(masked_rects, guided_thresh)
    if np.isfinite(masked_score):
        candidates.append({
            "mode": "masked",
            "score": masked_score,
            "reason": masked_reason,
            "rects": masked_rects,
            "thresh": guided_thresh,
        })

    if not candidates:
        return None

    best = max(candidates, key=lambda item: item["score"])

    return {
        "score": focus_row["score"],
        "region": masked_region if best["mode"] == "masked" else candidate_region,
        "rects": best["rects"],
        "thresh": best["thresh"],
        "color_mask": color_mask,
        "focus_mask": focus_row["mask"],
        "reason": f"{best['reason']} ({best['mode']})",
    }

def _detect_notebook_lines(enhanced):
    h_img, w_img = enhanced.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, int(w_img * 0.25)), 1))
    line_response = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    _, line_mask = cv2.threshold(line_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)),
    )

    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < w_img * 0.55 or h > max(30, int(h_img * 0.03)):
            continue
        lines.append((y, y + h))

    lines.sort()
    merged_lines = []
    merge_gap = max(10, int(h_img * 0.01))
    for y0, y1 in lines:
        if not merged_lines or y0 - merged_lines[-1][1] > merge_gap:
            merged_lines.append([y0, y1])
        else:
            merged_lines[-1][1] = max(merged_lines[-1][1], y1)

    return merged_lines, line_mask

def _filter_band_rects(rects, thresh_img, band_w, band_h):
    if not rects:
        return []

    min_area = max(20, int(band_h * band_w * 0.00012))
    stage_one = []
    for x, y, w, h in rects:
        area = w * h
        roi = thresh_img[max(0, y):y + h, max(0, x):x + w]
        fill_ratio = np.count_nonzero(roi) / max(1, area)

        if area < min_area or fill_ratio < 0.05:
            continue
        if w >= band_w * 0.45 and h <= max(18, int(band_h * 0.16)):
            continue
        if w > max(120, int(band_h * 1.0)):
            continue
        # More lenient height filtering to preserve thin digits
        min_h_strict = max(12, int(band_h * 0.15))
        min_h_lenient = max(18, int(band_h * 0.22))
        
        if h < min_h_strict and fill_ratio < 0.20:
            continue
        if h < min_h_lenient and fill_ratio < 0.15:
            continue
        if h > int(band_h * 0.92):
            continue

        stage_one.append((x, y, w, h))

    if not stage_one:
        return []

    med_w = float(np.median([rect[2] for rect in stage_one]))
    med_h = float(np.median([rect[3] for rect in stage_one]))
    med_a = float(np.median([rect[2] * rect[3] for rect in stage_one]))

    kept = []
    for x, y, w, h in stage_one:
        area = w * h
        touch_lr = x <= band_w * 0.03 or x + w >= band_w * (1.0 - 0.03)
        touch_tb = y <= band_h * 0.08 or y + h >= band_h * (1.0 - 0.08)

        if touch_tb and h >= med_h * 1.30 and area >= med_a * 1.30:
            continue
        if touch_lr and w >= med_w * 1.45 and h >= med_h * 1.15:
            continue
        if (touch_tb or touch_lr) and (w >= med_w * 1.85 or h >= med_h * 1.80):
            continue

        kept.append((x, y, w, h))

    return kept

def _keep_dominant_band_row(rects):
    if len(rects) <= 3:
        return rects

    row_snap = max(18, int(np.median([rect[3] for rect in rects]) * 0.65))
    centers = sorted((rect[1] + rect[3] / 2.0, index) for index, rect in enumerate(rects))

    groups = []
    current = [centers[0][1]]
    prev_center = centers[0][0]
    for center_y, index in centers[1:]:
        if center_y - prev_center > row_snap:
            groups.append(current)
            current = []
        current.append(index)
        prev_center = center_y
    groups.append(current)

    if len(groups) <= 1:
        return rects

    best_group = max(
        groups,
        key=lambda indices: (
            len(indices),
            sum(rects[index][2] * rects[index][3] for index in indices),
        ),
    )
    if len(best_group) < max(3, int(round(len(rects) * 0.5))):
        return rects

    return [rects[index] for index in sorted(best_group, key=lambda index: rects[index][0])]

def _validate_band_result(band_result, w_img, h_img):
    """
    Validate whether a band segmentation result is acceptable.
    
    A band result is considered VALID if:
    - number of rects >= 2 (minimum viable expression)
    - rect heights are consistent (std < 40% of median)
    - rects lie mostly on single row (use y-center clustering)
    - not dominated by thin line-like boxes
    - score is reasonable
    
    Returns: (is_valid, reason)
    """
    if band_result is None:
        return False, "No band result"
    
    score = band_result.get("score", 0)
    rects = band_result.get("rects", [])
    
    # Criterion 1: Must have at least 2 boxes
    if len(rects) < 2:
        return False, f"Too few rects ({len(rects)} < 2)"
    
    # Criterion 2: Must have reasonable score
    if score < 10.0:
        return False, f"Score too low ({score:.1f} < 10.0)"
    
    # Criterion 3: Heights should be reasonably consistent
    heights = [r[3] for r in rects]
    med_h = float(np.median(heights))
    std_h = float(np.std(heights))
    if std_h > med_h * 0.65:
        return False, f"Heights inconsistent (std/med={std_h/med_h:.2f} > 0.4)"
    
    # Criterion 4: Not dominated by line-like boxes
    line_like = sum(1 for rect in rects if _is_line_like_rect(rect, w_img, h_img))
    if line_like >= len(rects) * 0.5:
        return False, f"Line-dominated ({line_like}/{len(rects)})"
    
    # Criterion 5: Rects should be on single row (use y-center clustering)
    centers_y = sorted(r[1] + r[3] / 2.0 for r in rects)
    row_snap = max(18, int(med_h * 0.65))
    row_groups = 1
    for i in range(1, len(centers_y)):
        if centers_y[i] - centers_y[i-1] > row_snap:
            row_groups += 1
    if row_groups > 1:
        return False, f"Multiple rows detected ({row_groups} > 1)"
    
    return True, "Valid band result"

def _segment_notebook_band_fallback(enhanced, h_img, w_img):
    lines, line_mask = _detect_notebook_lines(enhanced)
    if len(lines) < 2:
        return None

    best = None
    for index in range(len(lines) - 1):
        gap = lines[index + 1][0] - lines[index][1]
        if gap < max(60, int(h_img * 0.05)) or gap > int(h_img * 0.40):
            continue

        pad = max(8, int(round(gap * 0.06)))
        y0 = max(0, lines[index][1] - pad)
        y1 = min(h_img, lines[index + 1][0] + pad)
        band = enhanced[y0:y1]
        band_h, band_w = band.shape
        if band_h < 32 or band_w < 32:
            continue

        block_size = min(51, max(21, (band_h // 2) | 1))
        limit = min(band_h, band_w)
        if block_size >= limit:
            block_size = max(21, limit - 1)
            if block_size % 2 == 0:
                block_size -= 1
        if block_size < 21:
            continue

        base = cv2.GaussianBlur(band, (9, 9), 0)
        for c_val in (-6, -8, -10):
            band_thresh = cv2.adaptiveThreshold(
                base,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_val,
            )
            band_thresh = cv2.medianBlur(band_thresh, 3)

            # Improved line removal: use larger kernel and more iterations for robustness
            line_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(60, int(band_w * 0.35)), 1),  # Larger width for thicker lines
            )
            band_lines = cv2.morphologyEx(band_thresh, cv2.MORPH_OPEN, line_kernel, iterations=1)
            band_lines = cv2.max(
                band_lines,
                _extract_horizontal_line_mask(
                    band_thresh,
                    min_span_ratio=0.34,
                    bridge_gap_ratio=0.08,
                    max_line_height_ratio=0.08,
                    lower_bias=False,
                ),
            )
            band_thresh = cv2.subtract(band_thresh, band_lines)
            # Additional vertical line removal if needed
            v_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(band_h * 0.15))))
            v_lines = cv2.morphologyEx(band_thresh, cv2.MORPH_OPEN, v_line_kernel, iterations=1)
            band_thresh = cv2.subtract(band_thresh, v_lines)

            band_thresh = cv2.morphologyEx(
                band_thresh,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )
            band_thresh = _remove_small_cc(
                band_thresh,
                max(12, int(band_h * band_w * 0.00005)),
            )

            # Preprocessing: light morphological opening to separate touching characters
            sep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            band_thresh = cv2.morphologyEx(band_thresh, cv2.MORPH_OPEN, sep_kernel, iterations=1)

            contours, _ = cv2.findContours(band_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = []
            for cnt in contours:
                if cv2.contourArea(cnt) < max(10.0, float(band_h * band_w) * 0.00004):
                    continue
                rects.append(cv2.boundingRect(cnt))

            rects = _filter_band_rects(rects, band_thresh, band_w, band_h)
            if rects:
                med_h = float(np.median([rect[3] for rect in rects]))
                med_w = float(np.median([rect[2] for rect in rects]))
                rects = _merge_broken_parts(rects, med_h, med_w)
                rects = _merge_single_digit_pairs(rects, band_thresh)
                rects = _remove_inside_boxes(rects)
                rects = _filter_band_rects(rects, band_thresh, band_w, band_h)
                rects = _keep_dominant_band_row(rects)

            if not rects:
                continue

            heights = [rect[3] for rect in rects]
            min_x = min(rect[0] for rect in rects)
            max_x = max(rect[0] + rect[2] for rect in rects)
            span_ratio = (max_x - min_x) / max(1.0, float(band_w))
            line_like = sum(
                1
                for rect in rects
                if _is_line_like_rect((rect[0], rect[1] + y0, rect[2], rect[3]), w_img, h_img)
            )

            score = 0.0
            score += len(rects) * 4.5
            score -= abs(len(rects) - 10) * 1.6
            score += min(4.0, span_ratio * 5.0)
            score += min(4.0, float(np.median(heights)) / max(1.0, float(band_h)) * 6.0)
            score -= line_like * 6.0

            if best is None or score > best["score"]:
                full_thresh = np.zeros((h_img, w_img), dtype=np.uint8)
                full_thresh[y0:y1, :] = band_thresh
                best = {
                    "score": score,
                    "thresh": full_thresh,
                    "rects": [(x, y + y0, w, h) for (x, y, w, h) in rects],
                    "line_mask": line_mask,
                }

    if best is None or best["score"] < 12.0:
        return None

    return best

def _recover_line_attached_blob_rects(binary, w_img, h_img, verbose=False):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recovered = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < max(72, int(w_img * 0.30)):
            continue
        if h < max(22, int(h_img * 0.06)):
            continue
        if w < max(h * 1.35, 28):
            continue

        roi = binary[y:y + h, x:x + w].copy()
        if roi.size == 0:
            continue

        line_mask = _extract_horizontal_line_mask(
            roi,
            min_span_ratio=0.42 if w >= 120 else 0.36,
            bridge_gap_ratio=0.10,
            max_line_height_ratio=0.18,
            lower_bias=True,
        )
        if np.count_nonzero(line_mask) == 0:
            lines = cv2.HoughLinesP(
                roi,
                1,
                np.pi / 180,
                threshold=max(24, int(w * 0.08)),
                minLineLength=max(28, int(w * 0.30)),
                maxLineGap=max(18, int(w * 0.10)),
            )
            if lines is None:
                continue

            thickness = max(2, int(round(h * 0.04)))
            for line in lines[:, 0]:
                x0, y0, x1, y1 = [int(v) for v in line]
                length = float(np.hypot(x1 - x0, y1 - y0))
                angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
                mid_y = (y0 + y1) / 2.0

                if abs(angle) > 10.0:
                    continue
                if length < w * 0.30:
                    continue
                if mid_y < h * 0.40:
                    continue

                cv2.line(line_mask, (x0, y0), (x1, y1), 255, thickness)

        if np.count_nonzero(line_mask) == 0:
            continue

        cleaned = cv2.subtract(roi, line_mask)
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
            iterations=1,
        )
        cleaned = _remove_small_cc(cleaned, max(6, int(w * h * 0.00035)))

        sub_rects = _collect_component_rects(cleaned, w, h)
        if len(sub_rects) >= 2:
            sub_rects = _split_wide_rects(sub_rects, cleaned)

        for sx, sy, sw, sh in sub_rects:
            area = sw * sh
            fill_ratio = _rect_fill_ratio((sx, sy, sw, sh), cleaned)
            if area < max(18, int(w * h * 0.00012)):
                continue
            if sh < max(14, int(h * 0.28)):
                continue
            if fill_ratio < 0.08:
                continue
            if sw >= w * 0.92 and sh >= h * 0.65:
                continue
            recovered.append((x + sx, y + sy, sw, sh))

    if verbose and recovered:
        print(f"[RECOVER] Restored {len(recovered)} rects from line-attached blobs")

    return sorted(recovered, key=lambda rect: rect[0])

def _collect_component_rects_with_line_recovery(binary, w_img, h_img):
    rects = _collect_component_rects(binary, w_img, h_img)
    recovered = _recover_line_attached_blob_rects(binary, w_img, h_img, verbose=False)
    if not recovered:
        return rects

    combined = list(rects)
    for rect in recovered:
        if any(_rect_overlap_ratio(rect, existing) >= 0.72 for existing in combined):
            continue
        combined.append(rect)

    return sorted(combined, key=lambda rect: rect[0])

def _recover_operator_component_rects(binary, existing_rects, w_img, h_img):
    if binary is None or getattr(binary, "size", 0) == 0:
        return []

    component_mask = (binary > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 8 or w < 2 or h < 3:
            continue
        components.append((x, y, w, h))

    if not components:
        return []

    reference_rects = existing_rects if existing_rects else components
    med_w = float(np.median([rect[2] for rect in reference_rects])) if reference_rects else 0.0
    med_h = float(np.median([rect[3] for rect in reference_rects])) if reference_rects else 0.0
    row_center = float(np.median([rect[1] + rect[3] / 2.0 for rect in reference_rects])) if reference_rects else (h_img / 2.0)

    recovered = []
    base_rects = existing_rects or []
    for rect in components:
        if any(_rect_overlap_ratio(rect, existing) >= 0.72 for existing in base_rects):
            continue

        x, y, w, h = rect
        fill_ratio = _rect_fill_ratio(rect, binary)
        aspect = w / max(1.0, float(h))
        cy = y + h / 2.0
        near_row = abs(cy - row_center) <= max(18.0, med_h * 1.05)

        dash_like = (
            near_row
            and w >= max(10.0, med_w * 0.22)
            and w <= max(42.0, med_w * 1.05)
            and h <= max(6.0, med_h * 0.18)
            and aspect >= 2.2
            and fill_ratio >= 0.18
        )
        slash_like = (
            near_row
            and x > 3
            and (x + w) < (w_img - 3)
            and h >= max(24.0, med_h * 1.20, h_img * 0.68)
            and w >= max(8.0, med_w * 0.18)
            and w <= max(28.0, med_w * 0.85)
            and aspect <= 0.42
            and 0.08 <= fill_ratio <= 0.40
        )

        if dash_like or slash_like:
            recovered.append(rect)

    return sorted(recovered, key=lambda rect: rect[0])

def _propose_expression_regions(thresh, w_img, h_img):
    rects = _collect_component_rects_with_line_recovery(thresh, w_img, h_img)
    if not rects:
        return [(0, 0, w_img, h_img)]

    heights = [r[3] for r in rects]
    widths = [r[2] for r in rects]
    med_h = float(np.median(heights)) if heights else 20.0
    med_w = float(np.median(widths)) if widths else 20.0
    row_snap = max(16, int(med_h * 0.85))

    charish = []
    for rect in rects:
        x, y, w, h = rect
        area = w * h
        aspect = w / max(1.0, float(h))
        if area < max(8, int(h_img * w_img * 0.00002)):
            continue
        if h < max(8, int(h_img * 0.018)):
            continue
        if h > h_img * 0.80 or w > w_img * 0.75:
            continue
        if aspect > 12.0 and h < med_h * 0.7:
            continue
        charish.append(rect)

    base_rects = charish if len(charish) >= 2 else rects
    med_a = float(np.median([r[2] * r[3] for r in base_rects])) if base_rects else 0.0
    margin_x = max(6, int(w_img * 0.02))
    margin_y = max(6, int(h_img * 0.02))
    core_rects = []
    for r in base_rects:
        x, y, w, h = r
        area = w * h
        touch_border = x <= margin_x or y <= margin_y or (x + w) >= (w_img - margin_x) or (y + h) >= (h_img - margin_y)
        if touch_border and area < max(12.0, med_a * 0.35):
            continue
        core_rects.append(r)
    group_source = core_rects if len(core_rects) >= 2 else base_rects

    centers = sorted((r[1] + r[3] / 2.0, idx) for idx, r in enumerate(group_source))

    groups = []
    current = [centers[0][1]] if centers else []
    prev_cy = centers[0][0] if centers else 0
    for cy, idx in centers[1:]:
        if cy - prev_cy > row_snap:
            groups.append(current)
            current = []
        current.append(idx)
        prev_cy = cy
    if current:
        groups.append(current)

    group_regions = []
    for indices in groups:
        group = [group_source[i] for i in indices]
        if len(group) == 0:
            continue
        xs = [r[0] for r in group]
        ys = [r[1] for r in group]
        x2s = [r[0] + r[2] for r in group]
        y2s = [r[1] + r[3] for r in group]
        pad_x = max(12, int(med_w * 0.9))
        pad_y = max(12, int(med_h * 0.9))
        x0 = max(0, min(xs) - pad_x)
        y0 = max(0, min(ys) - pad_y)
        x1 = min(w_img, max(x2s) + pad_x)
        y1 = min(h_img, max(y2s) + pad_y)
        if (x1 - x0) >= max(28, int(w_img * 0.08)) and (y1 - y0) >= max(24, int(h_img * 0.05)):
            group_regions.append((x0, y0, x1, y1))

    compact_region = None
    compact_source = core_rects if len(core_rects) >= 2 else base_rects
    if compact_source:
        xs = [r[0] for r in compact_source]
        ys = [r[1] for r in compact_source]
        x2s = [r[0] + r[2] for r in compact_source]
        y2s = [r[1] + r[3] for r in compact_source]
        pad_x = max(14, int(med_w * 1.15))
        pad_y = max(14, int(med_h * 1.10))
        compact_region = (
            max(0, min(xs) - pad_x),
            max(0, min(ys) - pad_y),
            min(w_img, max(x2s) + pad_x),
            min(h_img, max(y2s) + pad_y),
        )

    # Row-density proposals (generic, not notebook-specific)
    density_regions = []
    row_density = np.count_nonzero(thresh, axis=1).astype(np.float32)
    if row_density.size:
        k = max(9, ((h_img // 40) * 2 + 1))
        kernel = np.ones(k, dtype=np.float32) / float(k)
        smooth = np.convolve(row_density, kernel, mode='same')
        cutoff = max(float(np.mean(smooth) + 0.35 * np.std(smooth)), float(np.max(smooth) * 0.22))
        active = smooth >= cutoff
        start = None
        for i, flag in enumerate(active):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= max(20, int(h_img * 0.04)):
                    pad_y = max(10, int(med_h))
                    density_regions.append((0, max(0, start - pad_y), w_img, min(h_img, i + pad_y)))
                start = None
        if start is not None:
            pad_y = max(10, int(med_h))
            density_regions.append((0, max(0, start - pad_y), w_img, h_img))

    merged_groups = _merge_overlapping_regions(_dedupe_regions(group_regions + density_regions), w_img, h_img)
    candidates = []
    if compact_region is not None:
        candidates.append(compact_region)
    candidates.extend(group_regions)
    candidates.extend(density_regions)
    candidates.extend(merged_groups)

    full_region = (0, 0, w_img, h_img)
    candidates.append(full_region)
    return _dedupe_regions(candidates)

def _score_region_candidate(region, thresh, w_img, h_img):
    x0, y0, x1, y1 = region
    roi = thresh[y0:y1, x0:x1]
    if roi.size == 0:
        return float('-inf'), []

    rects = _collect_component_rects_with_line_recovery(roi, x1 - x0, y1 - y0)
    if not rects:
        return float('-inf'), []

    heights = [r[3] for r in rects]
    widths = [r[2] for r in rects]
    med_h = float(np.median(heights))
    med_w = float(np.median(widths))
    line_like = sum(1 for r in rects if r[2] >= (x1 - x0) * 0.28 and r[3] <= max(10, med_h * 0.45))
    span_x = (max(r[0] + r[2] for r in rects) - min(r[0] for r in rects)) / max(1.0, float(x1 - x0))
    y_centers = np.array([r[1] + r[3] / 2.0 for r in rects], dtype=np.float32)
    row_jitter = float(np.std(y_centers)) / max(1.0, med_h)
    fg_ratio = float(np.count_nonzero(roi)) / max(1.0, float(roi.size))
    region_aspect = (x1 - x0) / max(1.0, float(y1 - y0))
    region_frac = ((x1 - x0) * (y1 - y0)) / max(1.0, float(w_img * h_img))

    # Compactness: prefer regions where characters fill a meaningful span without being huge.
    min_x = min(r[0] for r in rects)
    max_x = max(r[0] + r[2] for r in rects)
    min_y = min(r[1] for r in rects)
    max_y = max(r[1] + r[3] for r in rects)
    tight_frac = ((max_x - min_x) * (max_y - min_y)) / max(1.0, float((x1 - x0) * (y1 - y0)))

    score = 0.0
    score += min(8.0, len(rects) * 1.5)
    score += min(4.0, span_x * 5.0)
    score += min(4.0, region_aspect * 0.6)
    score += min(4.0, tight_frac * 5.0)
    score -= min(6.0, line_like * 1.5)
    score -= abs(fg_ratio - 0.18) * 8.0
    score -= min(4.0, row_jitter * 2.5)
    score -= abs(med_w - med_h) / max(1.0, med_h) * 0.8
    score -= max(0.0, region_frac - 0.45) * 10.0

    full_rects = [(x + x0, y + y0, w, h) for x, y, w, h in rects]
    return score, full_rects

def _select_best_region(thresh, w_img, h_img):
    candidates = _propose_expression_regions(thresh, w_img, h_img)
    best_region = None
    best_score = float('-inf')
    best_rects = []
    for region in candidates:
        score, rects = _score_region_candidate(region, thresh, w_img, h_img)
        if score > best_score:
            best_score = score
            best_region = region
            best_rects = rects
    return best_region, best_score, best_rects

def _extend_region_along_row(thresh, region, w_img, h_img):
    """Expand a compact best-region horizontally to include plausible same-row characters."""
    if region is None:
        return None

    x0, y0, x1, y1 = map(int, region)
    rects = _collect_component_rects_with_line_recovery(thresh, w_img, h_img)
    if not rects:
        return region

    roi_h = max(1, y1 - y0)
    roi_cy = (y0 + y1) / 2.0

    seed_rects = []
    for r in rects:
        rx, ry, rw, rh = r
        rcx = rx + rw / 2.0
        rcy = ry + rh / 2.0
        intersects = not (rx + rw < x0 or rx > x1 or ry + rh < y0 or ry > y1)
        center_inside = (x0 <= rcx <= x1) and (y0 <= rcy <= y1)
        if intersects or center_inside:
            seed_rects.append(r)

    if not seed_rects:
        return region

    seed_heights = [r[3] for r in seed_rects]
    seed_widths = [r[2] for r in seed_rects]
    med_h = float(np.median(seed_heights)) if seed_heights else 12.0
    med_w = float(np.median(seed_widths)) if seed_widths else 10.0
    row_tol = max(12, int(max(roi_h, med_h) * 0.55))

    same_row = []
    for r in rects:
        rx, ry, rw, rh = r
        rcy = ry + rh / 2.0
        if abs(rcy - roi_cy) > row_tol:
            continue
        if rh < max(6.0, med_h * 0.45) or rh > max(h_img * 0.75, med_h * 2.3):
            continue
        if rw >= w_img * 0.45 and rh <= max(18.0, med_h * 0.75):
            continue
        same_row.append(r)

    if not same_row:
        return region

    same_row = sorted(same_row, key=lambda r: r[0])
    cluster = sorted(seed_rects[:], key=lambda r: r[0])
    med_w_row = float(np.median([r[2] for r in same_row])) if same_row else med_w
    max_gap = max(26, int(med_w_row * 3.2))

    changed = True
    while changed:
        changed = False
        left_edge = min(r[0] for r in cluster)
        right_edge = max(r[0] + r[2] for r in cluster)

        for r in same_row:
            if r in cluster:
                continue
            rx, ry, rw, rh = r
            gap_left = left_edge - (rx + rw)
            gap_right = rx - right_edge
            overlaps = not (rx + rw < left_edge or rx > right_edge)
            close_left = 0 <= gap_left <= max_gap
            close_right = 0 <= gap_right <= max_gap
            if overlaps or close_left or close_right:
                cluster.append(r)
                changed = True

        if changed:
            cluster = sorted(cluster, key=lambda r: r[0])

    xs = [r[0] for r in cluster]
    ys = [r[1] for r in cluster]
    x2s = [r[0] + r[2] for r in cluster]
    y2s = [r[1] + r[3] for r in cluster]

    pad_x = max(10, int(med_w_row * 0.9))
    pad_y = max(8, int(med_h * 0.6))
    ex0 = max(0, min(xs) - pad_x)
    ey0 = max(0, min(ys) - pad_y)
    ex1 = min(w_img, max(x2s) + pad_x)
    ey1 = min(h_img, max(y2s) + pad_y)

    ex0 = min(ex0, x0)
    ey0 = min(ey0, y0)
    ex1 = max(ex1, x1)
    ey1 = max(ey1, y1)

    return (int(ex0), int(ey0), int(ex1), int(ey1))

def _segment_generic_region(thresh, region, w_img, h_img):
    x0, y0, x1, y1 = region
    roi = thresh[y0:y1, x0:x1].copy()
    if roi.size == 0:
        return []

    roi_h, roi_w = roi.shape

    # Remove only very long ruling lines; keep short operators and digit '1'.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(36, int(roi_w * 0.62)), 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = _extract_vertical_guide_mask(
        roi,
        min_height_ratio=0.78,
        max_width_ratio=0.026,
        repeated_count=4,
    )
    roi = cv2.subtract(roi, cv2.add(h_lines, v_lines))

    raw_components = _collect_component_rects(roi, roi_w, roi_h)
    has_wide_attached_blob = any(
        rw >= max(72, int(roi_w * 0.28)) and rw >= max(rh * 1.35, 28)
        for _, _, rw, rh in raw_components
    )
    if has_wide_attached_blob:
        bridged_h_lines = _extract_horizontal_line_mask(
            roi,
            min_span_ratio=0.42,
            bridge_gap_ratio=0.10,
            max_line_height_ratio=0.18,
            lower_bias=True,
        )
        if np.count_nonzero(bridged_h_lines) > 0:
            roi = cv2.subtract(roi, bridged_h_lines)

    roi = cv2.medianBlur(roi, 3)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    roi = _remove_small_cc(roi, max(6, int(roi_h * roi_w * 0.00003)))

    recovered_rects = _recover_line_attached_blob_rects(roi, roi_w, roi_h, verbose=True)
    initial, merged, filtered = _extract_rects_from_thresh(roi, roi_w, roi_h)
    rects_local = filtered if filtered else merged if merged else initial
    if recovered_rects:
        if not rects_local:
            rects_local = list(recovered_rects)
        else:
            combined = list(rects_local)
            for rect in recovered_rects:
                if any(_rect_overlap_ratio(rect, existing) >= 0.72 for existing in combined):
                    continue
                combined.append(rect)
            rects_local = sorted(combined, key=lambda rect: rect[0])

    if not rects_local:
        rects_local = _recover_operator_component_rects(roi, [], roi_w, roi_h)
        if not rects_local:
            return []

    rects_local = _split_wide_rects(rects_local, roi)
    if rects_local:
        med_h = float(np.median([r[3] for r in rects_local]))
        med_w = float(np.median([r[2] for r in rects_local]))
        rects_local = _merge_broken_parts(sorted(rects_local, key=lambda r: r[0]), med_h, med_w)
        rects_local = _merge_single_digit_pairs(rects_local, roi)
        rects_local = _remove_inside_boxes(rects_local)

        operator_rects = _recover_operator_component_rects(roi, rects_local, roi_w, roi_h)
        if operator_rects:
            combined = list(rects_local)
            for rect in operator_rects:
                if any(_rect_overlap_ratio(rect, existing) >= 0.72 for existing in combined):
                    continue
                combined.append(rect)
            rects_local = sorted(combined, key=lambda rect: rect[0])

        rects_local = _operator_preserving_filter(rects_local, roi, roi_w, roi_h)
        rects_local = _keep_dominant_band_row(rects_local)

    return [(x + x0, y + y0, w, h) for x, y, w, h in rects_local]

def _validate_generic_result(rects, thresh, w_img, h_img):
    cfg = SEGMENTATION_CONFIG.generic_validation
    if rects is None or len(rects) == 0:
        return False, 'No rects found'
    if len(rects) < 2:
        return False, f'Too few rects ({len(rects)} < 2)'

    heights = [r[3] for r in rects]
    med_h = float(np.median(heights))
    if med_h <= 0:
        return False, 'Invalid median height'

    y_centers = sorted(r[1] + r[3] / 2.0 for r in rects)
    row_snap = max(cfg.row_snap_min, int(med_h * cfg.row_snap_ratio))
    rows = 1
    for i in range(1, len(y_centers)):
        if y_centers[i] - y_centers[i - 1] > row_snap:
            rows += 1
    if rows > cfg.max_rows:
        return False, f'Too many rows ({rows})'

    line_like = sum(1 for r in rects if _is_line_like_rect(r, w_img, h_img))
    if line_like >= max(cfg.min_line_like, int(round(len(rects) * cfg.line_like_ratio))):
        return False, f'Line dominated ({line_like}/{len(rects)})'

    widths = [r[2] for r in rects]
    ratios = [r[2] / max(1.0, float(r[3])) for r in rects]
    slim_ratio = float(np.mean(np.asarray(ratios, dtype=np.float32) < cfg.slim_rect_ratio)) if ratios else 0.0
    if len(rects) >= cfg.guide_dom_min_rects and slim_ratio >= cfg.guide_dom_ratio:
        return False, f'Vertical-guide dominated ({slim_ratio:.2f})'

    if np.std(widths) > max(cfg.width_std_abs, np.median(widths) * cfg.width_std_ratio):
        return False, 'Widths too inconsistent'

    return True, 'Valid generic region result'

def _score_upload_candidate(rects, thresh_img, w_img, h_img, region=None):
    cfg = SEGMENTATION_CONFIG.upload_candidate
    if rects is None or len(rects) < 2:
        return float("-inf"), {}
    if thresh_img is None or getattr(thresh_img, "size", 0) == 0:
        return float("-inf"), {}

    widths = np.asarray([r[2] for r in rects], dtype=np.float32)
    heights = np.asarray([r[3] for r in rects], dtype=np.float32)
    if widths.size == 0 or heights.size == 0:
        return float("-inf"), {}

    med_w = float(np.median(widths))
    med_h = float(np.median(heights))
    if med_w <= 0 or med_h <= 0:
        return float("-inf"), {}

    areas = widths * heights
    med_a = float(np.median(areas)) if areas.size else 0.0
    xs = np.asarray([r[0] for r in rects], dtype=np.float32)
    ys = np.asarray([r[1] for r in rects], dtype=np.float32)
    x2s = xs + widths
    y_centers = ys + heights / 2.0

    span_ratio = float((np.max(x2s) - np.min(xs)) / max(1.0, float(w_img)))
    row_jitter = float(np.std(y_centers) / max(1.0, med_h))
    width_cv = float(np.std(widths) / max(1.0, med_w))
    height_cv = float(np.std(heights) / max(1.0, med_h))
    aspect_ratio = float(med_w / max(1.0, med_h))
    rect_ratios = widths / np.maximum(1.0, heights)
    slim_ratio = float(np.mean(rect_ratios < cfg.slim_ratio_threshold))
    tiny_ratio = float(np.mean(
        (widths <= max(18.0, med_w * 0.42))
        | (heights <= max(24.0, med_h * 0.42))
        | (areas <= max(180.0, med_a * 0.28))
    ))
    fill_values = [_rect_fill_ratio(rect, thresh_img) for rect in rects]
    median_fill = float(np.median(fill_values)) if fill_values else 0.0
    line_like = sum(1 for rect in rects if _is_line_like_rect(rect, w_img, h_img))
    border_touch = sum(
        1
        for x, y, w, h in rects
        if x <= 1 or y <= 1 or (x + w) >= (w_img - 1) or (y + h) >= (h_img - 1)
    )

    region_frac = 0.0
    if region is not None:
        x0, y0, x1, y1 = region
        region_frac = float(max(0, x1 - x0) * max(0, y1 - y0)) / max(1.0, float(w_img * h_img))

    score = 0.0
    score += min(cfg.count_cap, len(rects) * cfg.count_weight)
    score += min(cfg.span_cap, span_ratio * cfg.span_weight)
    score += min(cfg.height_cap, (med_h / max(1.0, float(h_img))) * cfg.height_weight)
    score += max(0.0, cfg.row_bonus_base - row_jitter * cfg.row_bonus_weight)
    score += min(cfg.fill_cap, median_fill * cfg.fill_weight)
    score -= line_like * cfg.line_like_weight
    score -= tiny_ratio * cfg.tiny_ratio_weight
    score -= max(0.0, width_cv - cfg.width_cv_threshold) * cfg.width_cv_weight
    score -= max(0.0, height_cv - cfg.height_cv_threshold) * cfg.height_cv_weight
    score -= max(0.0, cfg.aspect_floor - aspect_ratio) * cfg.aspect_weight
    score -= max(0.0, slim_ratio - cfg.slim_ratio_threshold) * cfg.slim_ratio_weight
    score -= border_touch * cfg.border_touch_weight
    score -= max(0.0, region_frac - cfg.region_frac_threshold) * cfg.region_frac_weight

    metrics = {
        "count": int(len(rects)),
        "span_ratio": span_ratio,
        "row_jitter": row_jitter,
        "aspect_ratio": aspect_ratio,
        "slim_ratio": slim_ratio,
        "tiny_ratio": tiny_ratio,
        "median_fill": median_fill,
    }
    return score, metrics

def _select_best_upload_candidate(candidates, w_img, h_img):
    if not candidates:
        return None, []

    scored = []
    for candidate in candidates:
        selection_score, metrics = _score_upload_candidate(
            candidate.get("rects") or [],
            candidate.get("thresh"),
            w_img,
            h_img,
            region=candidate.get("region"),
        )
        if not np.isfinite(selection_score):
            continue

        enriched = dict(candidate)
        enriched["selection_score"] = float(selection_score)
        enriched["selection_metrics"] = metrics
        scored.append(enriched)

    if not scored:
        return None, []

    scored.sort(
        key=lambda item: (
            item["selection_score"],
            len(item.get("rects") or []),
            item["selection_metrics"].get("span_ratio", 0.0),
        ),
        reverse=True,
    )

    best = scored[0]
    region_candidate = next((item for item in scored if item.get("name") == "region"), None)
    if region_candidate is not None and best.get("name") != "region":
        rect_gain = len(region_candidate.get("rects") or []) - len(best.get("rects") or [])
        score_gap = float(best["selection_score"]) - float(region_candidate["selection_score"])
        best_span = float(best.get("selection_metrics", {}).get("span_ratio", 0.0))
        region_span = float(region_candidate.get("selection_metrics", {}).get("span_ratio", 0.0))
        if (
            rect_gain >= 3
            and score_gap <= 1.5
            and region_span >= best_span - 0.03
        ):
            best = region_candidate

    return best, scored

def _tighten_region_to_row_band(thresh_img, region, min_peak_ratio=0.25, pad_ratio=0.12):
    if thresh_img is None or getattr(thresh_img, "size", 0) == 0 or region is None:
        return region

    x0, y0, x1, y1 = [int(v) for v in region]
    roi = thresh_img[y0:y1, x0:x1]
    if roi.size == 0:
        return region

    row_counts = np.count_nonzero(roi, axis=1).astype(np.float32)
    if row_counts.size == 0:
        return region

    peak = float(np.max(row_counts))
    if peak <= 0:
        return region

    cutoff = max(5.0, peak * float(min_peak_ratio))
    active = np.where(row_counts >= cutoff)[0]
    if active.size == 0:
        return region

    row0 = int(active[0])
    row1 = int(active[-1])
    band_h = row1 - row0 + 1
    if band_h >= int(roi.shape[0] * 0.92):
        return region

    pad = max(12, int(round(band_h * float(pad_ratio))))
    new_y0 = max(y0, y0 + row0 - pad)
    new_y1 = min(y1, y0 + row1 + pad + 1)
    if new_y1 - new_y0 < max(24, int(roi.shape[0] * 0.18)):
        return region

    return (x0, new_y0, x1, new_y1)

def _extract_canvas_rects(thresh, w_img, h_img):
    initial, merged, filtered = _extract_rects_from_thresh(thresh, w_img, h_img)
    rects = filtered if filtered else merged if merged else initial
    if rects:
        rects = _operator_preserving_filter(rects, thresh, w_img, h_img)
    return initial, merged, rects
