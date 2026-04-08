import cv2
import numpy as np

from .logging_utils import get_logger, log_info_print

logger = get_logger(__name__)
print = lambda *args, **kwargs: log_info_print(*args, logger=logger, **kwargs)

def _should_merge(r1, r2, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    cx1 = x1 + w1 / 2
    cx2 = x2 + w2 / 2
    cy1 = y1 + h1 / 2
    cy2 = y2 + h2 / 2

    gap_x       = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    gap_y       = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    x_overlap   = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap   = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    combined_w  = max(x1 + w1, x2 + w2) - min(x1, x2)
    combined_h  = max(y1 + h1, y2 + h2) - min(y1, y2)
    small       = min(w1, w2) <= 14 or min(h1, h2) <= 14

    both_large = (w1 >= median_w * 0.50) and (w2 >= median_w * 0.50)

    max_combined = max(w1, w2, median_w) * 1.3
    if combined_w > max_combined:
        return False

    x_overlap_ratio = x_overlap / max(1.0, float(min(w1, w2)))
    center_dx = abs(cx1 - cx2)
    center_dy = abs(cy1 - cy2)

    stacked_fragment = (
        x_overlap_ratio >= 0.40
        and center_dx <= max(10.0, min(w1, w2) * 0.45, median_w * 0.35)
        and center_dy >= max(6.0, min(h1, h2) * 0.22)
        and gap_y <= max(18.0, median_h * 0.35)
        and combined_w <= max(w1, w2, median_w) * 1.40
        and combined_h <= max(median_h * 1.80, max(h1, h2) * 2.20)
    )
    if stacked_fragment:
        return True

    same_row_neighbors = (
        center_dy <= max(10.0, median_h * 0.22)
        and y_overlap >= min(h1, h2) * 0.45
    )
    if both_large and gap_x <= max(10, median_w * 0.18) and same_row_neighbors:
        return False

    vertically_close = (
        gap_y <= max(12, median_h * 0.18)
        and center_dy >= max(8.0, min(h1, h2) * 0.20)
        and x_overlap >= min(w1, w2) * 0.20
        and center_dx <= max(10.0, median_w * 0.45, min(w1, w2) * 0.60)
    )
    horizontally_close = (
        y_overlap > min(h1, h2) * 0.4
        and gap_x <= max(4, min(w1, w2) * 0.12)
        and combined_w <= max(w1, w2) * 1.6
    )
    small_neighbor = small and gap_x <= 12 and gap_y <= 12

    return vertically_close or horizontally_close or small_neighbor

def _merge_broken_parts(rects, median_h, median_w):
    """FIX-C: truyền median_w vào _should_merge."""
    if len(rects) < 2:
        return rects

    rects = sorted(rects, key=lambda r: r[0])
    used  = [False] * len(rects)
    merged = []

    for i in range(len(rects)):
        if used[i]:
            continue
        x, y, w, h = rects[i]
        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if used[j] or i == j:
                    continue
                if _should_merge((x, y, w, h), rects[j], median_h, median_w):
                    xj, yj, wj, hj = rects[j]
                    nx = min(x, xj)
                    ny = min(y, yj)
                    x, y, w, h = nx, ny, max(x+w, xj+wj)-nx, max(y+h, yj+hj)-ny
                    used[j] = True
                    changed  = True
                    print(f"[MERGE] Gop manh X={xj} vao cum X={x}")
                    break
        merged.append((x, y, w, h))

    return merged

def _rect_is_inside(inner, outer, pad=0):
    x1, y1, w1, h1 = inner
    x2, y2, w2, h2 = outer
    return (
        x1 >= (x2 - pad)
        and y1 >= (y2 - pad)
        and (x1 + w1) <= (x2 + w2 + pad)
        and (y1 + h1) <= (y2 + h2 + pad)
    )

def _remove_inside_boxes(rects, thresh_img=None, w_img=None, h_img=None):
    if len(rects) < 2:
        return rects

    widths = [r[2] for r in rects]
    heights = [r[3] for r in rects]
    areas = [r[2] * r[3] for r in rects]
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))
    med_a = float(np.median(areas))

    remove_indices = set()

    for i, rect_i in enumerate(rects):
        x1, y1, w1, h1 = rect_i
        area_i = w1 * h1

        contained = [
            j
            for j, rect_j in enumerate(rects)
            if i != j and _rect_is_inside(rect_j, rect_i, pad=1)
        ]
        containers = [
            j
            for j, rect_j in enumerate(rects)
            if i != j and _rect_is_inside(rect_i, rect_j, pad=1)
        ]

        if contained:
            contained_area = sum(rects[j][2] * rects[j][3] for j in contained)
            fill_ratio = _rect_fill_ratio(rect_i, thresh_img) if thresh_img is not None else 0.0
            wrapper_like = (
                len(contained) >= 2
                and (
                    w1 >= max(72.0, med_w * 1.60)
                    or h1 >= max(72.0, med_h * 1.40)
                    or area_i >= max(1800.0, med_a * 2.00)
                )
                and (
                    contained_area >= area_i * 0.22
                    or fill_ratio <= 0.28
                )
            )
            if wrapper_like:
                remove_indices.add(i)
                print(f"[INSIDE] Xoa wrapper box tai X={x1}")
                continue

        if containers:
            outer_areas = [rects[j][2] * rects[j][3] for j in containers]
            smallest_outer = min(outer_areas) if outer_areas else float("inf")
            tiny_inside = (
                area_i <= min(max(900.0, med_a * 0.45), smallest_outer * 0.38)
                and (
                    w1 <= max(18.0, med_w * 0.42)
                    or h1 <= max(24.0, med_h * 0.42)
                )
            )
            if tiny_inside:
                remove_indices.add(i)
                print(f"[INSIDE] Xoa hop nho nam trong hop lon tai X={x1}")

    return [
        rect
        for idx, rect in enumerate(rects)
        if idx not in remove_indices
    ]

def _filter_rects(rects, thresh_img, w_img, h_img):
    """
    FIX-3: Ngưỡng area tương đối (scale theo median).
    FIX-B: Lọc bleed-through bằng fill_ratio tuyệt đối thấp.

    Bleed-through (chữ thấm qua trang, Image 1) có đặc điểm:
    - Bounding box vừa phải (không nhỏ, không lớn)
    - fill_ratio rất thấp (< 0.06) vì nét mờ, đứt đoạn nhiều
    - Thường nằm ở hàng khác với số chính
    Lọc chúng bằng cách chặn fill_ratio < 0.06 với box không quá nhỏ.
    """
    if not rects:
        return []

    widths  = [r[2] for r in rects]
    heights = [r[3] for r in rects]
    areas   = [r[2] * r[3] for r in rects]
    med_w   = float(np.median(widths))
    med_h   = float(np.median(heights))
    med_a   = float(np.median(areas))

    min_area_abs = max(6, int(h_img * w_img * 0.00004))
    min_area_rel = med_a * 0.08

    margin_x = max(6, int(w_img * 0.02))
    margin_y = max(6, int(h_img * 0.02))

    kept = []
    for (x, y, w, h) in rects:
        area         = w * h
        aspect       = w / max(1.0, float(h))
        inv_aspect   = h / max(1.0, float(w))
        roi          = thresh_img[max(0,y):y+h, max(0,x):x+w]
        fill_ratio   = np.count_nonzero(roi) / max(1, area)
        on_border    = (x <= margin_x or y <= margin_y
                        or x + w >= w_img - margin_x
                        or y + h >= h_img - margin_y)

        # FIX-3: ngưỡng area tương đối
        if area < min_area_abs and area < min_area_rel:
            print(f"[-] Qua nho tai X={x} (area={area})")
            continue

        # FIX-B: lọc bleed-through — fill rất thấp + box không phải quá nhỏ
        # (box nhỏ đã bị lọc ở trên; box vừa mà fill thấp = bleed-through)
        if (
            fill_ratio < 0.06
            and area > min_area_abs * 2
            and not _looks_like_sparse_seven_box((x, y, w, h), thresh_img, med_h, med_w)
        ):
            print(f"[-] Bleed-through tai X={x} (fill={fill_ratio:.3f})")
            continue

        # Đường ngang dài còn sót
        if (
            aspect >= 6.0
            and w >= max(w_img * 0.18, med_w * 1.4)
            and h <= max(18.0, med_h * 0.65)
            and fill_ratio <= 0.55
        ):
            print(f"[-] Duong ngang con sot tai X={x}")
            continue

        # Noise nhỏ ở viền
        small_border_blob = (
            area < max(min_area_abs, med_a * 0.12)
            and w < med_w * 0.75
            and h < med_h * 0.75
        )
        if on_border and small_border_blob:
            print(f"[-] Nhieu vien tai X={x}")
            continue

        # Nét viền mỏng dài
        if on_border and fill_ratio <= 0.35 and (
            (aspect >= 4.5 and w >= med_w * 1.2)
            or (inv_aspect >= 4.5 and h >= med_h * 1.2)
        ):
            print(f"[-] Net mong vien tai X={x}")
            continue

        kept.append((x, y, w, h))

    return kept

def _extract_rects_from_thresh(thresh, w_img, h_img):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(6.0, float(h_img * w_img) * 0.00002)

    initial = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < w_img * 0.95 and h < h_img * 0.95 and h >= 5 and w >= 2:
            initial.append((x, y, w, h))

    if initial:
        med_h = float(np.median([r[3] for r in initial]))
        med_w = float(np.median([r[2] for r in initial]))
    else:
        med_h, med_w = 20.0, 20.0

    merged = _merge_broken_parts(initial, med_h, med_w)
    merged = _remove_inside_boxes(merged)
    filtered = _filter_rects(merged, thresh, w_img, h_img)
    return initial, merged, filtered

def _is_line_like_rect(rect, w_img, h_img):
    _, _, w, h = rect
    return w >= w_img * 0.18 and h <= max(18, int(h_img * 0.03))

def _is_line_dominated(rects, w_img, h_img):
    if not rects:
        return True

    line_like = sum(1 for rect in rects if _is_line_like_rect(rect, w_img, h_img))
    median_w = float(np.median([rect[2] for rect in rects]))
    median_h = float(np.median([rect[3] for rect in rects]))
    median_aspect = float(np.median([rect[2] / max(1.0, float(rect[3])) for rect in rects]))

    return (
        line_like >= max(3, int(round(len(rects) * 0.4)))
        or (
            len(rects) >= 5
            and median_w >= w_img * 0.25
            and median_h <= max(18.0, h_img * 0.04)
            and median_aspect >= 5.5
        )
    )

def _crop_with_padding(thresh_img, x, y, w, h, pad_ratio=0.15):
    pad  = max(4, int(round(max(w, h) * pad_ratio)))
    H, W = thresh_img.shape
    roi  = thresh_img[max(0, y-pad):min(H, y+h+pad),
                      max(0, x-pad):min(W, x+w+pad)]
    return roi if roi.size > 0 else None

def _sort_by_row_col(rects, row_snap=None):
    if not rects:
        return []

    if row_snap is None:
        heights   = [r[3] for r in rects]
        row_snap  = max(10, int(np.median(heights) * 0.6))

    centers_y = [(r[1] + r[3] / 2, i) for i, r in enumerate(rects)]
    centers_y.sort()

    row_labels = [0] * len(rects)
    current_row = 0
    prev_cy = centers_y[0][0]

    for cy, idx in centers_y:
        if cy - prev_cy > row_snap:
            current_row += 1
        row_labels[idx] = current_row
        prev_cy = cy

    labeled = [(row_labels[i], rects[i][0], rects[i]) for i in range(len(rects))]
    labeled.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in labeled]

def _rect_fill_ratio(rect, thresh_img):
    x, y, w, h = rect
    roi = thresh_img[max(0, y):y + h, max(0, x):x + w]
    return float(np.count_nonzero(roi)) / max(1.0, float(w * h))

def _count_connected_components_in_roi(roi):
    if roi is None or getattr(roi, "size", 0) == 0:
        return 0

    binary = (roi > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    count = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 6:
            count += 1
    return count

def _looks_like_sparse_seven_box(rect, thresh_img, median_h, median_w):
    x, y, w, h = rect
    roi = thresh_img[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return False

    fill_ratio = np.count_nonzero(roi) / max(1.0, float(roi.size))
    aspect = w / max(1.0, float(h))
    binary = (roi > 0).astype(np.uint8)
    top = binary[:max(1, h // 4), :]
    bottom = binary[h // 2:, :]
    top_density = np.count_nonzero(top) / max(1.0, float(top.size))
    bottom_density = np.count_nonzero(bottom) / max(1.0, float(bottom.size))
    component_count = _count_connected_components_in_roi(roi)

    return (
        max(14.0, median_w * 0.60) <= w <= max(76.0, median_w * 1.25)
        and h >= max(80.0, median_h * 1.15)
        and 0.22 <= aspect <= 0.58
        and 0.04 <= fill_ratio <= 0.11
        and top_density >= 0.12
        and bottom_density <= 0.08
        and 2 <= component_count <= 3
    )

def _count_holes_in_binary_roi(roi):
    if roi is None or getattr(roi, "size", 0) == 0:
        return 0

    binary = np.where(roi > 0, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0

    holes = 0
    for idx, node in enumerate(hierarchy[0]):
        parent = int(node[3])
        if parent == -1:
            continue
        if cv2.contourArea(contours[idx]) >= 6.0:
            holes += 1
    return holes

def _rect_overlap_ratio(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    ix0 = max(x1, x2)
    iy0 = max(y1, y2)
    ix1 = min(x1 + w1, x2 + w2)
    iy1 = min(y1 + h1, y2 + h2)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter = float((ix1 - ix0) * (iy1 - iy0))
    return inter / max(1.0, float(min(w1 * h1, w2 * h2)))

def _should_merge_digit_pair(r1, r2, thresh_img, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x2 < x1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    gap_x = max(0, x2 - (x1 + w1))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    height_ratio = min(h1, h2) / max(1.0, float(max(h1, h2)))
    combined_x0 = min(x1, x2)
    combined_y0 = min(y1, y2)
    combined_x1 = max(x1 + w1, x2 + w2)
    combined_y1 = max(y1 + h1, y2 + h2)
    combined_w = combined_x1 - combined_x0
    combined_h = combined_y1 - combined_y0
    aspect = combined_w / max(1.0, float(combined_h))
    slim_fragment_pair = (
        max(w1, w2) <= max(42.0, median_h * 0.40)
        and combined_w <= max(68.0, median_h * 0.78)
    )

    if gap_x > max(3.0, median_w * 0.08, median_h * 0.05):
        return False
    if y_overlap < min(h1, h2) * 0.55:
        return False
    if height_ratio < 0.72:
        return False

    roi = thresh_img[combined_y0:combined_y1, combined_x0:combined_x1]
    hole_count = _count_holes_in_binary_roi(roi)
    fill_ratio = np.count_nonzero(roi) / max(1.0, float(roi.size))

    split_loop_pair = (
        hole_count == 1
        and max(w1, w2) <= max(120.0, median_w * 1.20)
        and combined_w <= max(180.0, median_w * 1.75)
        and combined_h <= max(190.0, median_h * 1.35)
    )

    if not (slim_fragment_pair or split_loop_pair):
        return False

    if not (0.52 <= aspect <= 1.18):
        return False

    if split_loop_pair:
        if fill_ratio < 0.14 or fill_ratio > 0.48:
            return False
        return True

    if hole_count < 1:
        return False
    if fill_ratio < 0.10 or fill_ratio > 0.42:
        return False

    return True

def _should_merge_topbar_tail_pair(r1, r2, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if y2 < y1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    gap_x = max(0, x2 - (x1 + w1))
    vertical_gap = max(0, y2 - (y1 + h1))
    combined_w = max(x1 + w1, x2 + w2) - min(x1, x2)
    combined_h = max(y1 + h1, y2 + h2) - min(y1, y2)

    top_bar_like = (
        h1 <= max(18.0, median_h * 0.22)
        and w1 >= max(22.0, median_w * 0.70)
    )
    lower_tail_like = (
        w2 <= max(14.0, median_w * 0.25)
        and h2 <= max(42.0, median_h * 0.40)
    )

    return (
        top_bar_like
        and lower_tail_like
        and gap_x <= max(18.0, median_w * 0.32)
        and vertical_gap >= max(20.0, median_h * 0.40)
        and combined_w <= max(72.0, median_w * 1.25)
        and combined_h <= max(190.0, median_h * 1.80)
    )

def _should_merge_satellite_fragment_pair(r1, r2, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x2 < x1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    area1 = w1 * h1
    area2 = w2 * h2
    if area1 <= area2:
        sx, sy, sw, sh = x1, y1, w1, h1
        lx, ly, lw, lh = x2, y2, w2, h2
    else:
        sx, sy, sw, sh = x2, y2, w2, h2
        lx, ly, lw, lh = x1, y1, w1, h1

    small_area = sw * sh
    large_area = lw * lh
    gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    combined_w = max(x1 + w1, x2 + w2) - min(x1, x2)
    combined_h = max(y1 + h1, y2 + h2) - min(y1, y2)
    small_center_y = sy + (sh / 2.0)
    large_center_y = ly + (lh / 2.0)
    near_left_side = sx + sw <= lx + max(8.0, lw * 0.38)
    near_right_side = sx >= lx + lw - max(8.0, lw * 0.38)

    return (
        small_area <= max(900.0, large_area * 0.35)
        and sw <= max(42.0, median_w * 0.68)
        and sh <= max(52.0, median_h * 0.46)
        and gap_x <= max(18.0, median_w * 0.32)
        and y_overlap >= sh * 0.45
        and abs(small_center_y - large_center_y) <= max(28.0, lh * 0.38)
        and (near_left_side or near_right_side)
        and combined_w <= max(150.0, median_w * 2.35)
        and combined_h <= max(lh * 1.15, median_h * 1.15)
    )

def _should_merge_short_left_with_stroke_pair(r1, r2, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x2 < x1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    gap_x = max(0, x2 - (x1 + w1))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    combined_w = (x2 + w2) - x1
    combined_h = max(y1 + h1, y2 + h2) - min(y1, y2)

    return (
        gap_x <= max(26.0, median_w * 0.42)
        and h1 <= h2 * 0.72
        and w1 >= max(28.0, w2 * 0.75)
        and w1 <= max(52.0, median_w * 0.82)
        and w2 <= max(44.0, median_w * 0.70)
        and h2 >= max(92.0, median_h * 0.90)
        and y_overlap >= h1 * 0.82
        and y1 >= y2 + max(6.0, h2 * 0.05)
        and combined_w <= max(112.0, median_w * 1.78)
        and combined_h <= max(150.0, median_h * 1.25)
    )

def _should_merge_staggered_fragment_pair(r1, r2, thresh_img, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x2 < x1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    gap_x = max(0, x2 - (x1 + w1))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    center_y_delta = abs((y1 + h1 / 2.0) - (y2 + h2 / 2.0))
    combined_x0 = min(x1, x2)
    combined_y0 = min(y1, y2)
    combined_x1 = max(x1 + w1, x2 + w2)
    combined_y1 = max(y1 + h1, y2 + h2)
    combined_w = combined_x1 - combined_x0
    combined_h = combined_y1 - combined_y0

    if gap_x > max(10.0, median_w * 0.20):
        return False
    if y_overlap < min(h1, h2) * 0.18:
        return False
    if center_y_delta < max(18.0, median_h * 0.28):
        return False
    if combined_w > max(92.0, median_w * 2.10):
        return False
    if combined_h > max(180.0, median_h * 2.00):
        return False

    max_part_w = max(w1, w2)
    min_part_h = min(h1, h2)
    if max_part_w > max(72.0, median_w * 1.15):
        return False
    if min_part_h < max(26.0, median_h * 0.45):
        return False

    roi = thresh_img[combined_y0:combined_y1, combined_x0:combined_x1]
    if roi.size == 0:
        return False

    fill_ratio = np.count_nonzero(roi) / max(1.0, float(roi.size))
    if fill_ratio < 0.07 or fill_ratio > 0.46:
        return False

    closed = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    if _count_connected_components_in_roi(closed) > 3:
        return False

    return True

def _should_merge_offset_satellite_pair(r1, r2, thresh_img, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x2 < x1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    area1 = w1 * h1
    area2 = w2 * h2
    if area1 <= area2:
        sx, sy, sw, sh = x1, y1, w1, h1
        lx, ly, lw, lh = x2, y2, w2, h2
        small_area, large_area = area1, area2
    else:
        sx, sy, sw, sh = x2, y2, w2, h2
        lx, ly, lw, lh = x1, y1, w1, h1
        small_area, large_area = area2, area1

    gap_x = max(0, x2 - (x1 + w1))
    center_y_delta = abs((y1 + h1 / 2.0) - (y2 + h2 / 2.0))
    combined_x0 = min(x1, x2)
    combined_y0 = min(y1, y2)
    combined_x1 = max(x1 + w1, x2 + w2)
    combined_y1 = max(y1 + h1, y2 + h2)
    combined_w = combined_x1 - combined_x0
    combined_h = combined_y1 - combined_y0
    near_left_side = sx + sw <= lx + max(12.0, lw * 0.28)
    near_right_side = sx >= lx + lw - max(12.0, lw * 0.28)

    if small_area > max(850.0, large_area * 0.28):
        return False
    if sw > max(32.0, median_w * 0.55):
        return False
    if sh > max(62.0, median_h * 0.65):
        return False
    if gap_x > max(42.0, median_w * 0.95):
        return False
    if center_y_delta > max(28.0, median_h * 0.38):
        return False
    if combined_w > max(124.0, median_w * 2.10):
        return False
    if combined_h > max(170.0, median_h * 1.35):
        return False
    if not (near_left_side or near_right_side):
        return False

    roi = thresh_img[combined_y0:combined_y1, combined_x0:combined_x1]
    if roi.size == 0:
        return False

    fill_ratio = np.count_nonzero(roi) / max(1.0, float(roi.size))
    if fill_ratio < 0.04 or fill_ratio > 0.48:
        return False

    closed = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    if _count_connected_components_in_roi(closed) > 4:
        return False

    return True

def _should_merge_inline_fragment_pair(r1, r2, thresh_img, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x2 < x1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    area1 = w1 * h1
    area2 = w2 * h2
    if area1 <= area2:
        sx, sy, sw, sh = x1, y1, w1, h1
        lx, ly, lw, lh = x2, y2, w2, h2
        small_area, large_area = area1, area2
    else:
        sx, sy, sw, sh = x2, y2, w2, h2
        lx, ly, lw, lh = x1, y1, w1, h1
        small_area, large_area = area2, area1

    gap_x = max(0, x2 - (x1 + w1))
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    vertical_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    center_y_delta = abs((y1 + h1 / 2.0) - (y2 + h2 / 2.0))
    combined_x0 = min(x1, x2)
    combined_y0 = min(y1, y2)
    combined_x1 = max(x1 + w1, x2 + w2)
    combined_y1 = max(y1 + h1, y2 + h2)
    combined_w = combined_x1 - combined_x0
    combined_h = combined_y1 - combined_y0

    if combined_w > max(118.0, median_w * 1.58):
        return False
    if combined_h > max(190.0, median_h * 1.45):
        return False

    touching_inline = (
        gap_x <= max(8.0, median_w * 0.12)
        or x_overlap >= max(3.0, min(w1, w2) * 0.15)
    )
    if not touching_inline:
        return False

    if vertical_gap > max(18.0, median_h * 0.18):
        return False
    if center_y_delta > max(34.0, median_h * 0.36):
        return False

    small_fragment_like = (
        small_area <= max(1100.0, large_area * 0.42)
        or sw <= max(28.0, median_w * 0.40)
        or sh <= max(34.0, median_h * 0.34)
    )
    if not small_fragment_like:
        return False

    # Không gộp nếu cả 2 ký tự đều cao (tương đương với chữ số đầy đủ hoặc dấu ngoặc '(')
    if h1 >= median_h * 0.8 and h2 >= median_h * 0.8 and gap_x >= max(2.0, median_w * 0.05):
        return False

    # Avoid merging a genuine narrow digit such as "1" with its neighbor.
    small_full_digit_like = (
        sh >= max(72.0, median_h * 0.82)
        and sw >= max(18.0, median_w * 0.26)
        and small_area >= max(1500.0, median_h * median_w * 0.22)
        and x_overlap <= max(2.0, sw * 0.10)
        and gap_x >= max(4.0, median_w * 0.06)
    )
    if small_full_digit_like:
        return False

    near_x_edge = (
        sx + sw <= lx + max(10.0, lw * 0.34)
        or sx >= lx + lw - max(10.0, lw * 0.34)
        or (sx >= lx and sx + sw <= lx + lw)
    )
    near_y_edge = (
        sy + sh <= ly + max(12.0, lh * 0.34)
        or sy >= ly + lh - max(12.0, lh * 0.34)
        or (sy >= ly and sy + sh <= ly + lh)
    )
    if not (near_x_edge or near_y_edge):
        return False

    roi = thresh_img[combined_y0:combined_y1, combined_x0:combined_x1]
    if roi.size == 0:
        return False

    fill_ratio = np.count_nonzero(roi) / max(1.0, float(roi.size))
    if fill_ratio < 0.05 or fill_ratio > 0.55:
        return False

    hole_count = _count_holes_in_binary_roi(roi)
    if hole_count > 2:
        return False

    closed = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    if _count_connected_components_in_roi(closed) > 3:
        return False

    return y_overlap >= min(h1, h2) * 0.12 or vertical_gap <= max(8.0, median_h * 0.08)

def _should_merge_corner_fragment_pair(r1, r2, thresh_img, median_h, median_w):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x2 < x1:
        x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1

    area1 = w1 * h1
    area2 = w2 * h2
    if area1 <= area2:
        sx, sy, sw, sh = x1, y1, w1, h1
        lx, ly, lw, lh = x2, y2, w2, h2
        small_area, large_area = area1, area2
    else:
        sx, sy, sw, sh = x2, y2, w2, h2
        lx, ly, lw, lh = x1, y1, w1, h1
        small_area, large_area = area2, area1

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    gap_x = max(0, x2 - (x1 + w1))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    vertical_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    combined_x0 = min(x1, x2)
    combined_y0 = min(y1, y2)
    combined_x1 = max(x1 + w1, x2 + w2)
    combined_y1 = max(y1 + h1, y2 + h2)
    combined_w = combined_x1 - combined_x0
    combined_h = combined_y1 - combined_y0

    if combined_w > max(32.0, median_w * 1.62):
        return False
    if combined_h > max(194.0, median_h * 1.55):
        return False

    if small_area > max(1000.0, large_area * 0.34):
        return False
    if sw > max(24.0, median_w * 0.34):
        return False
    if sh > max(48.0, median_h * 0.52):
        return False

    touching_horizontally = (
        x_overlap >= max(4.0, sw * 0.22)
        or gap_x <= max(8.0, median_w * 0.10)
    )
    if not touching_horizontally:
        return False

    if y_overlap > sh * 0.55:
        return False
    if vertical_gap > max(32.0, median_h * 0.28):
        return False

    near_top = sy + sh <= ly + max(18.0, lh * 0.34)
    near_bottom = sy >= ly + lh - max(18.0, lh * 0.34)
    if not (near_top or near_bottom):
        return False

    roi = thresh_img[combined_y0:combined_y1, combined_x0:combined_x1]
    if roi.size == 0:
        return False

    fill_ratio = np.count_nonzero(roi) / max(1.0, float(roi.size))
    if fill_ratio < 0.05 or fill_ratio > 0.50:
        return False

    closed = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    if _count_connected_components_in_roi(closed) > 4:
        return False

    return True

def _merge_rect_pair(r1, r2):
    x0 = min(r1[0], r2[0])
    y0 = min(r1[1], r2[1])
    x1 = max(r1[0] + r1[2], r2[0] + r2[2])
    y1 = max(r1[1] + r1[3], r2[1] + r2[3])
    return (x0, y0, x1 - x0, y1 - y0)

def _merge_single_digit_pairs(rects, thresh_img):
    if len(rects) < 2:
        return rects

    current_rects = sorted(rects, key=lambda r: r[0])
    max_passes = min(4, max(1, len(current_rects) - 1))

    for _ in range(max_passes):
        median_h = float(np.median([r[3] for r in current_rects]))
        median_w = float(np.median([r[2] for r in current_rects]))
        merged = []
        i = 0
        changed = False

        while i < len(current_rects):
            current = current_rects[i]
            if i + 1 < len(current_rects):
                nxt = current_rects[i + 1]
                if (
                    _should_merge_digit_pair(current, nxt, thresh_img, median_h, median_w)
                    or _should_merge_topbar_tail_pair(current, nxt, median_h, median_w)
                    or _should_merge_satellite_fragment_pair(current, nxt, median_h, median_w)
                    or _should_merge_short_left_with_stroke_pair(current, nxt, median_h, median_w)
                    or _should_merge_staggered_fragment_pair(current, nxt, thresh_img, median_h, median_w)
                    or _should_merge_offset_satellite_pair(current, nxt, thresh_img, median_h, median_w)
                    or _should_merge_inline_fragment_pair(current, nxt, thresh_img, median_h, median_w)
                    or _should_merge_corner_fragment_pair(current, nxt, thresh_img, median_h, median_w)
                ):
                    merged.append(_merge_rect_pair(current, nxt))
                    changed = True
                    i += 2
                    continue
            merged.append(current)
            i += 1

        current_rects = sorted(merged, key=lambda r: r[0])
        if not changed:
            break

    return current_rects

def _collect_component_rects(binary, w_img, h_img):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(6.0, float(h_img * w_img) * 0.000015)
    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= w_img * 0.98 or h >= h_img * 0.98:
            continue
        if w < 2 or h < 4:
            continue
        rects.append((x, y, w, h))
    return rects

def _merge_overlapping_regions(regions, w_img, h_img):
    if not regions:
        return []

    pad_x = max(10, int(w_img * 0.02))
    pad_y = max(10, int(h_img * 0.02))
    merged = []
    for x0, y0, x1, y1 in sorted(regions):
        if not merged:
            merged.append([x0, y0, x1, y1])
            continue
        lx0, ly0, lx1, ly1 = merged[-1]
        overlap_x = min(x1, lx1) - max(x0, lx0)
        overlap_y = min(y1, ly1) - max(y0, ly0)
        near_x = x0 <= lx1 + pad_x and x1 >= lx0 - pad_x
        near_y = y0 <= ly1 + pad_y and y1 >= ly0 - pad_y
        if overlap_x > -pad_x and overlap_y > -pad_y and near_x and near_y:
            merged[-1] = [min(lx0, x0), min(ly0, y0), max(lx1, x1), max(ly1, y1)]
        else:
            merged.append([x0, y0, x1, y1])
    return [tuple(r) for r in merged]

def _dedupe_regions(regions, min_area=64):
    unique = []
    seen = set()
    for x0, y0, x1, y1 in regions:
        x0 = int(max(0, x0))
        y0 = int(max(0, y0))
        x1 = int(max(x0 + 1, x1))
        y1 = int(max(y0 + 1, y1))
        area = (x1 - x0) * (y1 - y0)
        if area < min_area:
            continue
        key = (x0, y0, x1, y1)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique

def _tight_bbox_from_cols(base_x, base_y, roi_part):
    """Tính tight bounding box từ ROI binary, trả về tọa độ tuyệt đối."""
    cols = np.where(np.count_nonzero(roi_part, axis=0) > 0)[0]
    rows = np.where(np.count_nonzero(roi_part, axis=1) > 0)[0]
    if cols.size == 0 or rows.size == 0:
        return None
    return (
        base_x + int(cols[0]),
        base_y + int(rows[0]),
        max(2, int(cols[-1]) - int(cols[0]) + 1),
        max(2, int(rows[-1]) - int(rows[0]) + 1),
    )

def _find_split_valleys(proj, width, valley_threshold=0.40, margin=0.18):
    """
    Tìm tất cả valley hợp lệ trong vertical projection histogram.

    Trả về list các (valley_x, valley_score) đã sắp xếp theo score tăng dần
    (score thấp = valley sâu hơn = tốt hơn để cắt).

    Args:
        proj       : 1-D float array — số pixel trắng theo từng cột
        width      : chiều rộng ROI (== len(proj))
        valley_threshold : tỉ lệ valley/peak_ref tối đa để xem là valid cut
        margin     : bỏ qua vùng biên (mỗi bên margin*width)
    """
    if proj.size < 7:
        return []

    smooth_k = max(3, (width // 18) * 2 + 1)
    kernel = np.ones(smooth_k, dtype=np.float32) / float(smooth_k)
    smooth = np.convolve(proj, kernel, mode="same")

    left  = int(width * margin)
    right = int(width * (1.0 - margin))
    if right - left < 3:
        return []

    zone = smooth[left:right]
    peak_global = float(np.max(smooth))
    if peak_global <= 0:
        return []

    # Tìm local minima trong vùng giữa
    valleys = []
    for i in range(1, len(zone) - 1):
        if zone[i] <= zone[i - 1] and zone[i] <= zone[i + 1]:
            abs_i = i + left
            # peak reference = max của nửa trái và nửa phải valley
            left_peak  = float(np.max(smooth[:abs_i]))  if abs_i > 0 else 0.0
            right_peak = float(np.max(smooth[abs_i + 1:])) if abs_i + 1 < len(smooth) else 0.0
            peak_ref   = max(left_peak, right_peak, 1.0)
            score      = float(zone[i]) / peak_ref
            if score <= valley_threshold:
                valleys.append((abs_i, score))

    # Sắp xếp theo score tăng dần (valley sâu nhất lên đầu)
    valleys.sort(key=lambda v: v[1])
    return valleys

def _split_roi_at_valleys(roi, base_x, base_y, valleys, min_segment_w=4):
    """
    Cắt ROI tại các vị trí valley đã cho, trả về list bounding boxes.
    Chỉ giữ lại các segment có pixel > 0.
    """
    if not valleys:
        return None

    # Sắp xếp cut points theo x
    cut_xs = sorted(set(v[0] for v in valleys))
    segments = []
    prev = 0
    for cx in cut_xs:
        if cx - prev >= min_segment_w:
            segments.append((prev, cx))
        prev = cx
    if roi.shape[1] - prev >= min_segment_w:
        segments.append((prev, roi.shape[1]))

    if len(segments) < 2:
        return None

    result = []
    for seg_x0, seg_x1 in segments:
        seg = roi[:, seg_x0:seg_x1]
        if np.count_nonzero(seg) == 0:
            continue
        bbox = _tight_bbox_from_cols(base_x + seg_x0, base_y, seg)
        if bbox is not None:
            result.append(bbox)

    return result if len(result) >= 2 else None

def _estimate_char_count(w, med_w, med_h=None):
    """Ước lượng số ký tự trong một box dựa theo tỉ lệ chiều rộng.

    FIX: Khi med_w bị inflate do các boxes gộp (chữ to viết gần nhau),
    dùng med_h làm reference thứ hai vì chữ viết tay thường gần vuông.
    Lấy estimate nhỏ hơn trong hai cách để tránh oversplit.
    """
    if med_w <= 0:
        return 1
    ratio = w / float(med_w)

    if med_h is not None and med_h > 0:
        # Relaxed constraint: Only apply ratio_h cap if the box is extraordinarily short, which would imply it's not multiple normal characters. Otherwise trust ratio.
        ratio_h = w / float(med_h)
        if ratio_h < 0.6:
            ratio = min(ratio, ratio_h * 1.5)

    if ratio < 1.55:
        return 1
    if ratio < 2.45:
        return 2
    if ratio < 3.45:
        return 3
    return int(round(ratio))

def _cc_split(roi, base_x, base_y, med_h, min_area_ratio=0.01):
    """
    Fallback: erode nhẹ rồi lấy connected components để cắt.
    Hữu ích khi hai ký tự chạm nhau nhưng projection không tạo valley rõ ràng.
    """
    sep = cv2.erode(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sep, connectivity=8)
    min_area = max(6, int(roi.shape[0] * roi.shape[1] * min_area_ratio))
    comps = []
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        sh   = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area < min_area or sh < max(6, int(med_h * 0.4)):
            continue
        comps.append((
            int(stats[lbl, cv2.CC_STAT_LEFT]),
            int(stats[lbl, cv2.CC_STAT_TOP]),
            int(stats[lbl, cv2.CC_STAT_WIDTH]),
            int(stats[lbl, cv2.CC_STAT_HEIGHT]),
        ))
    comps.sort(key=lambda r: r[0])
    if len(comps) < 2:
        return None

    result = []
    for sx, sy, sw, sh in comps:
        part = roi[sy:sy + sh, sx:sx + sw]
        bbox = _tight_bbox_from_cols(base_x + sx, base_y + sy, part)
        if bbox is not None:
            result.append(bbox)
    return result if len(result) >= 2 else None

def _split_single_rect(rect, thresh_img, med_w, med_h):
    """
    Tách 1 rect rộng thành N ký tự nhỏ hơn.

    Chiến lược:
    1. Ước tính số ký tự kỳ vọng (n_expected).
    2. Tìm tất cả valleys trong projection histogram.
    3. Nếu valleys đủ (>= n_expected - 1): chọn n_expected - 1 valleys sâu nhất.
    4. Nếu không đủ valley: thử CC-split.
    5. Fallback: chia đều (worst case).
    6. Đệ quy: nếu segment nào vẫn còn rộng, split tiếp.

    Trả về list bounding boxes, hoặc [rect] nếu không split được.
    """
    x, y, w, h = rect
    roi = thresh_img[y:y + h, x:x + w]
    if roi.size == 0 or w < 8:
        return [rect]

    # FIX: truyền med_h vào để estimate chính xác hơn khi med_w bị inflate
    n_expected = _estimate_char_count(w, med_w, med_h)
    if n_expected < 2:
        return [rect]

    n_cuts = n_expected - 1

    # ── Bước 1: Projection-valley split ──────────────────────────────────────
    proj    = np.count_nonzero(roi, axis=0).astype(np.float32)
    # FIX: valley_threshold được tăng lên 0.58 để bắt được các valley nông giữa 2 chữ dính nhau
    valleys = _find_split_valleys(proj, w, valley_threshold=0.58, margin=0.15)

    split_result = None
    if len(valleys) >= n_cuts:
        # Chọn n_cuts valley sâu nhất (score nhỏ nhất)
        chosen_valleys = valleys[:n_cuts]
        split_result = _split_roi_at_valleys(roi, x, y, chosen_valleys)

    # ── Bước 2: CC-split fallback ─────────────────────────────────────────────
    if split_result is None and n_expected == 2:
        split_result = _cc_split(roi, x, y, med_h)

    # ── Bước 3: Uniform split (worst-case) ────────────────────────────────────
    if split_result is None and w >= max(med_w * 1.7, med_h * 0.85):
        seg_w = w // n_expected
        if seg_w >= 4:
            parts = []
            for k in range(n_expected):
                seg_x0 = k * seg_w
                seg_x1 = seg_x0 + seg_w if k < n_expected - 1 else w
                seg = roi[:, seg_x0:seg_x1]
                if np.count_nonzero(seg) == 0:
                    continue
                bbox = _tight_bbox_from_cols(x + seg_x0, y, seg)
                if bbox is not None:
                    parts.append(bbox)
            if len(parts) >= 2:
                split_result = parts

    if split_result is None:
        return [rect]

    # ── Bước 4: Đệ quy trên từng segment vẫn còn rộng ────────────────────────
    final = []
    for seg_rect in split_result:
        sx, sy, sw, sh = seg_rect
        if sw >= max(med_w * 1.8, med_h * 1.1) and sh >= med_h * 0.7:
            sub = _split_single_rect(seg_rect, thresh_img, med_w, med_h)
            final.extend(sub)
        else:
            final.append(seg_rect)

    return final if final else [rect]

def _split_wide_rects(rects, thresh_img):
    print(f"[SPLIT_WIDE] Input {len(rects)} rects, widths={sorted([r[2] for r in rects])}")
    """
    Thay thế _split_wide_rects cũ bằng valley-based recursive split.

    Cải tiến so với phiên bản trước:
    - Tìm nhiều valley thay vì chỉ lấy 1 điểm thấp nhất ở giữa
    - Đệ quy: box sau khi cắt vẫn còn rộng sẽ được cắt tiếp
    - Ước tính số ký tự kỳ vọng để chọn đúng số valley
    - CC-split và uniform-split làm fallback có thứ tự rõ ràng

    FIX: Dùng IQR (interquartile range) thay cho median thuần để tính med_w.
    Khi có boxes gộp (2 số dính nhau), chúng sẽ có width >> normal,
    kéo median lên cao → estimate char count thấp → không split đủ.
    IQR loại trừ các outlier lớn này, cho med_w sát với ký tự đơn hơn.
    """
    if len(rects) < 2:
        return rects

    # FIX: dùng IQR để tính med_w ổn định hơn khi có boxes gộp
    widths = sorted([r[2] for r in rects])
    n = len(widths)
    q1_idx = max(0, n // 4)
    q3_idx = min(n - 1, 3 * n // 4)
    iqr_widths = widths[q1_idx:q3_idx + 1]
    med_w = float(np.median(iqr_widths)) if iqr_widths else float(np.median(widths))

    med_h = float(np.median([r[3] for r in rects]))
    out   = []

    for rect in sorted(rects, key=lambda r: r[0]):
        x, y, w, h = rect
        # FIX: width_threshold không nên phụ thuộc vào med_h vì 1 box có thể rất rộng bất kể chiều cao
        if w < med_w * 1.70 or h < med_h * 0.50:
            out.append(rect)
            continue

        split_result = _split_single_rect(rect, thresh_img, med_w, med_h)
        out.extend(split_result)
        if len(split_result) > 1:
            print(f"[SPLIT] Box X={x} w={w} -> {len(split_result)} segments (med_w={med_w:.1f}, med_h={med_h:.1f})")

    return out

def _operator_preserving_filter(rects, thresh_img, w_img, h_img):
    if not rects:
        return []

    base = _filter_rects(rects, thresh_img, w_img, h_img)
    kept = {tuple(r) for r in base}
    med_h = float(np.median([r[3] for r in rects]))
    med_w = float(np.median([r[2] for r in rects]))
    centers = [r[1] + r[3] / 2.0 for r in rects]
    row_center = float(np.median(centers)) if centers else h_img / 2.0
    
    for rect in rects:
        if tuple(rect) in kept:
            continue
        x, y, w, h = rect
        area = w * h
        fill_ratio = _rect_fill_ratio(rect, thresh_img)
        cy = y + h / 2.0
        near_row = abs(cy - row_center) <= max(12.0, med_h * 0.9)

        dash_like = (
            w >= max(6.0, med_w * 0.35)
            and w <= max(28.0, med_w * 1.8)
            and h <= max(12.0, med_h * 0.45)
            and fill_ratio >= 0.18
        )
        dot_like = (
            area >= 4
            and w <= max(10.0, med_w * 0.5)
            and h <= max(10.0, med_h * 0.5)
            and fill_ratio >= 0.12
        )
        slim_operator = (
            w <= max(14.0, med_w * 0.7)
            and h >= max(10.0, med_h * 0.45)
            and h <= max(28.0, med_h * 1.4)
            and fill_ratio >= 0.12
        )

        if near_row and (dash_like or dot_like or slim_operator):
            kept.add(tuple(rect))

    return sorted([tuple(r) for r in kept], key=lambda r: r[0])
