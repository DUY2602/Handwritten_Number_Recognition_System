import cv2
import numpy as np


TERM_END_CHARS = set("0123456789)")
TERM_START_CHARS = set("0123456789(")
OPERATORS = set("+-*/")


def _safe_mean(arr):
    return float(arr.mean()) if arr.size else 0.0


def _side_balance(features):
    left = max(1e-6, features["left"])
    right = max(1e-6, features["right"])
    return max(left, right) / min(left, right)


def _content_crop(roi):
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        h, w = roi.shape[:2]
        return roi.astype(np.float32), (0, 0, w, h)

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return roi[y0:y1, x0:x1].astype(np.float32), (x0, y0, x1 - x0, y1 - y0)


def _extract_features(roi, rect):
    crop, content_box = _content_crop(roi)
    binary = (crop > 0).astype(np.float32)
    binary_u8 = (binary * 255).astype(np.uint8)
    ch, cw = binary.shape[:2]

    half_h = max(1, ch // 2)
    half_w = max(1, cw // 2)
    third_h = max(1, ch // 3)
    third_w = max(1, cw // 3)

    tl = binary[:half_h, :half_w]
    tr = binary[:half_h, cw - half_w:]
    bl = binary[ch - half_h:, :half_w]
    br = binary[ch - half_h:, cw - half_w:]
    center = binary[third_h:ch - third_h, third_w:cw - third_w]
    center_row = binary[max(0, ch // 2 - 1):min(ch, ch // 2 + 2), :]
    center_col = binary[:, max(0, cw // 2 - 1):min(cw, cw // 2 + 2)]

    holes = 0
    contours, hierarchy = cv2.findContours(binary_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for node in hierarchy[0]:
            if node[3] != -1:
                holes += 1

    return {
        "rect_area": float(rect[2] * rect[3]),
        "rect_w": float(rect[2]),
        "rect_h": float(rect[3]),
        "y_center": float(rect[1]) + float(rect[3]) / 2.0,
        "content_box": content_box,
        "content_w": float(cw),
        "content_h": float(ch),
        "aspect": float(cw) / max(1.0, float(ch)),
        "holes": int(holes),
        "fill": _safe_mean(binary),
        "left": _safe_mean(binary[:, :half_w]),
        "right": _safe_mean(binary[:, cw - half_w:]),
        "top": _safe_mean(binary[:half_h, :]),
        "bottom": _safe_mean(binary[ch - half_h:, :]),
        "center": _safe_mean(center),
        "center_row": _safe_mean(center_row),
        "center_col": _safe_mean(center_col),
        "diag_primary": _safe_mean(tl) + _safe_mean(br),
        "diag_secondary": _safe_mean(tr) + _safe_mean(bl),
    }


def _looks_like_minus(features):
    return (
        features["aspect"] >= 2.0
        and features["content_h"] <= 6
        and features["center_row"] >= 0.45
    )


def _looks_like_slash(features):
    diagonal_margin = features["diag_secondary"] - features["diag_primary"]
    return (
        features["holes"] == 0
        and features["content_h"] >= 12
        and features["aspect"] <= 1.05
        and features["fill"] <= 0.42
        and features["center"] <= 0.78
        and features["center_col"] <= 0.52
        and features["diag_secondary"] >= 0.72
        and diagonal_margin >= 0.42
    )


def _looks_like_lparen(features):
    return (
        features["holes"] == 0
        and features["content_h"] >= 14
        and features["aspect"] <= 0.65
        and 0.25 <= features["fill"] <= 0.78
        and features["center"] >= 0.35
        and features["center_row"] >= 0.42
        and 0.42 <= features["center_col"] <= 0.90
        and features["left"] >= features["right"] * 1.35
    )


def _looks_like_rparen(features):
    return (
        features["holes"] == 0
        and features["content_h"] >= 14
        and features["aspect"] <= 0.65
        and 0.25 <= features["fill"] <= 0.78
        and features["center"] >= 0.35
        and features["center_row"] >= 0.42
        and 0.42 <= features["center_col"] <= 0.90
        and features["right"] >= features["left"] * 1.35
    )


def _looks_like_open_bracketish(features):
    return (
        features["holes"] == 0
        and features["content_h"] >= 14
        and features["aspect"] <= 0.42
        and 0.24 <= features["fill"] <= 0.72
        and features["center"] <= 0.12
        and features["center_row"] <= 0.38
        and features["center_col"] <= 0.36
        and features["top"] >= 0.22
        and features["bottom"] >= 0.22
        and features["left"] >= features["right"] * 2.10
    )


def _looks_like_draw_open_boundary(features):
    return (
        features["holes"] == 0
        and features["content_h"] >= 14
        and features["aspect"] <= 0.40
        and 0.34 <= features["fill"] <= 0.62
        and features["center"] <= 0.36
        and features["center_row"] <= 0.43
        and features["center_col"] <= 0.55
        and features["top"] >= 0.34
        and features["bottom"] >= 0.34
        and features["left"] >= features["right"] * 1.75
    )


def _looks_like_close_bracketish(features):
    return (
        features["holes"] == 0
        and features["content_h"] >= 14
        and features["aspect"] <= 0.42
        and 0.24 <= features["fill"] <= 0.72
        and features["center"] <= 0.12
        and features["center_row"] <= 0.38
        and features["center_col"] >= 0.48
        and features["top"] >= 0.22
        and features["bottom"] >= 0.22
        and features["right"] >= features["left"] * 2.10
    )


def _looks_like_open_boundary(features):
    return (
        _looks_like_lparen(features)
        or _looks_like_open_bracketish(features)
        or _looks_like_draw_open_boundary(features)
    )


def _looks_like_close_boundary(features):
    return _looks_like_rparen(features) or _looks_like_close_bracketish(features)


def _looks_like_star(features):
    return (
        features["holes"] == 0
        and 0.75 <= features["aspect"] <= 1.2
        and features["fill"] >= 0.45
        and features["center"] >= 0.75
        and features["diag_primary"] >= 1.15
        and features["diag_secondary"] >= 1.15
        and abs(features["diag_primary"] - features["diag_secondary"]) <= 0.2
    )


def _looks_like_straight_one(features):
    return (
        features["holes"] == 0
        and features["content_h"] >= 14
        and features["aspect"] <= 0.9
        and features["fill"] >= 0.28
        and features["center"] >= 0.45
        and features["center_col"] >= 0.5
        and features["content_w"] >= 6
        and _side_balance(features) <= 2.1
    )


def _looks_like_slanted_one(features):
    diagonal_margin = abs(features["diag_secondary"] - features["diag_primary"])
    return (
        features["holes"] == 0
        and features["content_h"] >= 18
        and features["aspect"] <= 0.60
        and 0.20 <= features["fill"] <= 0.46
        and 0.24 <= features["center"] <= 0.72
        and features["center_row"] >= 0.30
        and features["center_col"] <= 0.40
        and features["content_w"] <= 12
        and _side_balance(features) >= 1.65
        and diagonal_margin >= 0.55
    )


def _looks_like_one(features):
    return _looks_like_straight_one(features) or _looks_like_slanted_one(features)


def _looks_like_three_override(features):
    return (
        features["holes"] <= 1
        and 0.70 <= features["aspect"] <= 1.00
        and features["center"] <= 0.18
        and features["center_row"] >= 0.16
        and features["center_col"] <= 0.24
        and features["right"] >= features["left"] * 1.25
        and features["bottom"] >= features["top"] * 0.95
    )


def _looks_like_sparse_seven_override(features):
    return (
        features["holes"] == 0
        and 0.30 <= features["aspect"] <= 0.58
        and 0.08 <= features["fill"] <= 0.20
        and features["top"] >= 0.18
        and features["bottom"] <= 0.06
        and features["center"] <= 0.06
        and features["center_col"] <= 0.22
    )


def _looks_like_slash_override(features):
    diagonal_margin = features["diag_secondary"] - features["diag_primary"]
    return (
        _looks_like_slash(features)
        and 0.65 <= features["aspect"] <= 1.0
        and features["fill"] <= 0.36
        and features["center"] <= 0.70
        and features["center_col"] <= 0.40
        and features["center_row"] <= 0.58
        and features["diag_secondary"] >= 0.82
        and diagonal_margin >= 0.52
    )


def _looks_like_strong_slash_override(features):
    diagonal_margin = features["diag_secondary"] - features["diag_primary"]
    return (
        0.72 <= features["aspect"] <= 0.9
        and features["holes"] == 0
        and features["fill"] <= 0.31
        and features["center"] <= 0.57
        and features["center_row"] <= 0.30
        and features["center_col"] <= 0.36
        and features["diag_secondary"] >= 0.94
        and diagonal_margin >= 0.68
    )


def _looks_like_thin_slash_override(features):
    diagonal_margin = features["diag_secondary"] - features["diag_primary"]
    return (
        features["holes"] == 0
        and 0.30 <= features["aspect"] <= 0.55
        and features["fill"] <= 0.32
        and features["center_row"] <= 0.34
        and features["center_col"] <= 0.36
        and abs(features["left"] - features["right"]) <= 0.08
        and features["diag_secondary"] >= 1.00
        and diagonal_margin >= 0.75
    )


def _is_open_paren_context(prev_char, next_char):
    prev_ok = prev_char is None or prev_char in OPERATORS or prev_char == "("
    next_ok = next_char in TERM_START_CHARS or next_char == "-"
    return prev_ok and next_ok


def _is_close_paren_context(prev_char, next_char):
    prev_ok = prev_char in TERM_END_CHARS
    next_ok = next_char is None or next_char in OPERATORS or next_char == ")"
    return prev_ok and next_ok


def _top_k_score(item, target_char):
    entries = item.get("top_k") or []
    for entry in entries:
        if str(entry.get("char")) == target_char:
            return float(entry.get("conf", 0.0))

    if not entries and str(item.get("raw_char", item.get("char", ""))) == target_char:
        return float(item.get("raw_conf", item.get("conf", 0.0)))

    return 0.0


def _has_model_support(item, target_char, min_conf=0.08, allow_when_missing=True):
    entries = item.get("top_k") or []
    if not entries:
        return allow_when_missing

    if str(item.get("raw_char", item.get("char", ""))) == target_char:
        return True

    return _top_k_score(item, target_char) >= min_conf


def _has_near_top_model_support(
    item,
    target_char,
    min_conf=0.25,
    max_gap=0.08,
    allow_when_missing=False,
):
    entries = item.get("top_k") or []
    if not entries:
        return allow_when_missing

    if str(item.get("raw_char", item.get("char", ""))) == target_char:
        return True

    target_score = _top_k_score(item, target_char)
    if target_score < min_conf:
        return False

    best_score = max(float(entry.get("conf", 0.0)) for entry in entries)
    return (best_score - target_score) <= max_gap


def _should_drop_noise(item, median_area, median_height):
    features = item["features"]
    if item["conf"] < 0.55 and features["rect_area"] < max(25.0, median_area * 0.1):
        return True
    if (
        features["rect_area"] < max(120.0, median_area * 0.02)
        and features["rect_h"] < max(14.0, median_height * 0.2)
    ):
        return True
    if features["content_h"] < 5 and item["conf"] < 0.98:
        return True
    if (
        item["char"] == "-"
        and item["conf"] < 0.995
        and features["rect_area"] < max(32.0, median_area * 0.22)
        and features["rect_h"] < max(10.0, median_height * 0.35)
    ):
        return True
    return False


def _looks_like_numeric_edge_fragment(item, median_area, median_height):
    features = item["features"]
    small = (
        features["rect_area"] <= max(180.0, median_area * 0.32)
        and features["rect_h"] <= max(32.0, median_height * 0.46)
    )
    operatorish = (
        item["char"] in OPERATORS
        or item["char"] in {"(", ")"}
        or _looks_like_minus(features)
        or _looks_like_slash(features)
    )
    return small and operatorish


def _drop_numeric_edge_noise(items, median_area, median_height):
    if len(items) < 6:
        return items

    trimmed = list(sorted(items, key=lambda item: item["rect"][0]))
    changed = True
    while changed and len(trimmed) >= 6:
        changed = False

        lead = trimmed[0]
        rest = trimmed[1:]
        if _looks_like_numeric_edge_fragment(lead, median_area, median_height) and _looks_like_numeric_line(rest):
            trimmed = rest
            changed = True
            continue

        tail = trimmed[-1]
        rest = trimmed[:-1]
        if _looks_like_numeric_edge_fragment(tail, median_area, median_height) and _looks_like_numeric_line(rest):
            trimmed = rest
            changed = True

    return trimmed


def _build_combined_numeric_item(left_item, right_item):
    lx, ly, lw, lh = left_item["rect"]
    rx, ry, rw, rh = right_item["rect"]
    x0 = min(lx, rx)
    y0 = min(ly, ry)
    x1 = max(lx + lw, rx + rw)
    y1 = max(ly + lh, ry + rh)

    left_crop, _ = _content_crop(left_item["roi"])
    right_crop, _ = _content_crop(right_item["roi"])
    if left_crop.size == 0 or right_crop.size == 0:
        return None

    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    left_scaled = cv2.resize(
        (left_crop > 0).astype(np.uint8) * 255,
        (max(1, lw), max(1, lh)),
        interpolation=cv2.INTER_NEAREST,
    )
    right_scaled = cv2.resize(
        (right_crop > 0).astype(np.uint8) * 255,
        (max(1, rw), max(1, rh)),
        interpolation=cv2.INTER_NEAREST,
    )

    lxs = lx - x0
    lys = ly - y0
    rxs = rx - x0
    rys = ry - y0
    canvas[lys:lys + lh, lxs:lxs + lw] = np.maximum(canvas[lys:lys + lh, lxs:lxs + lw], left_scaled)
    canvas[rys:rys + rh, rxs:rxs + rw] = np.maximum(canvas[rys:rys + rh, rxs:rxs + rw], right_scaled)

    combined_rect = (x0, y0, x1 - x0, y1 - y0)
    combined_features = _extract_features(canvas, combined_rect)
    return canvas, combined_rect, combined_features


def _classify_combined_numeric_pair(features):
    if features["holes"] >= 2:
        return "8"

    if features["holes"] == 1:
        if features["right"] >= features["left"] * 1.15 and features["center"] >= 0.14:
            return "9"
        if features["left"] >= features["right"] * 1.15 and features["center"] <= 0.12:
            return "6"
        return "0"

    if (
        0.34 <= features["aspect"] <= 0.60
        and features["right"] >= features["left"] * 1.60
        and features["center"] <= 0.16
        and features["center_col"] <= 0.15
        and features["diag_primary"] >= features["diag_secondary"] * 2.20
    ):
        return "9"

    return None


def _looks_like_split_numeric_fragment(item, median_height):
    features = item["features"]
    raw_char = str(item.get("raw_char", item.get("char", "")))

    narrow_width = features["rect_w"] <= max(36.0, median_height * 0.40)
    narrow_aspect = features["aspect"] <= 0.42
    partial_height = features["rect_h"] <= max(84.0, median_height * 0.92)
    slender_label = raw_char in {"1", "7", "/", "\\", "(", ")"}

    return partial_height and (slender_label or narrow_width or narrow_aspect)


def _merge_numeric_split_pairs(items, median_height):
    if len(items) < 2:
        return items

    merged = []
    ordered = list(sorted(items, key=lambda item: item["rect"][0]))
    i = 0
    while i < len(ordered):
        current = ordered[i]
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            lx, ly, lw, lh = current["rect"]
            rx, ry, rw, rh = nxt["rect"]
            gap = max(0, rx - (lx + lw))
            y_overlap = max(0, min(ly + lh, ry + rh) - max(ly, ry))
            combined_width = max(lx + lw, rx + rw) - min(lx, rx)
            looks_like_fragments = (
                _looks_like_split_numeric_fragment(current, median_height)
                and _looks_like_split_numeric_fragment(nxt, median_height)
            )

            if (
                current.get("roi") is not None
                and nxt.get("roi") is not None
                and looks_like_fragments
                and gap <= max(8.0, median_height * 0.18)
                and y_overlap >= min(lh, rh) * 0.32
                and combined_width <= max(60.0, median_height * 0.72)
            ):
                combined = _build_combined_numeric_item(current, nxt)
                if combined is not None:
                    combined_roi, combined_rect, combined_features = combined
                    combined_char = _classify_combined_numeric_pair(combined_features)
                    combined_aspect = combined_rect[2] / max(1.0, float(combined_rect[3]))
                    if combined_char is not None and combined_aspect <= 1.15:
                        merged.append({
                            "char": combined_char,
                            "conf": max(float(current["conf"]), float(nxt["conf"])),
                            "raw_char": f"{current['raw_char']}{nxt['raw_char']}",
                            "raw_conf": max(float(current["raw_conf"]), float(nxt["raw_conf"])),
                            "top_k": [],
                            "rect": combined_rect,
                            "features": combined_features,
                            "roi": combined_roi,
                            "adjusted": True,
                        })
                        i += 2
                        continue

        merged.append(current)
        i += 1

    return merged


def _cluster_lines(items, median_height):
    tolerance = max(28.0, median_height * 0.65)
    groups = []

    for item in sorted(items, key=lambda candidate: candidate["features"]["y_center"]):
        y_center = item["features"]["y_center"]
        placed = False

        for group in groups:
            if abs(y_center - group["center"]) <= tolerance:
                group["items"].append(item)
                group["center"] = float(np.mean([
                    member["features"]["y_center"] for member in group["items"]
                ]))
                placed = True
                break

        if not placed:
            groups.append({"center": y_center, "items": [item]})

    return [group["items"] for group in groups]


def _filter_main_line(items, median_area, median_height):
    core_items = [
        item for item in items
        if item["features"]["rect_area"] >= max(150.0, median_area * 0.2)
    ]
    if len(core_items) < 3:
        return sorted(items, key=lambda item: item["rect"][0])

    y_centers = [item["features"]["y_center"] for item in core_items]
    line_spread = max(y_centers) - min(y_centers)
    if line_spread <= max(120.0, median_height * 1.65):
        return sorted(items, key=lambda item: item["rect"][0])

    line_groups = [group for group in _cluster_lines(core_items, median_height) if len(group) >= 3]
    if len(line_groups) <= 1:
        return sorted(items, key=lambda item: item["rect"][0])

    def _line_score(group):
        count_score = len(group) * 1000.0
        area_score = sum(item["features"]["rect_area"] for item in group)
        y_score = -min(item["features"]["y_center"] for item in group)
        return count_score + area_score + y_score

    selected_group = max(line_groups, key=_line_score)
    selected_center = float(np.mean([item["features"]["y_center"] for item in selected_group]))
    tolerance = max(40.0, median_height * 0.85)

    filtered = [
        item for item in items
        if abs(item["features"]["y_center"] - selected_center) <= tolerance
    ]
    return sorted(filtered or items, key=lambda item: item["rect"][0])


def _looks_like_numeric_line(items):
    if len(items) < 5:
        return False

    strong_ops = 0
    for item in items:
        char = item["char"]
        conf = item["conf"]
        features = item["features"]

        if char == "*" and conf >= 0.9:
            strong_ops += 1
        elif char in {"+", "4"} and conf >= 0.9 and _looks_like_star(features):
            strong_ops += 1
        elif char == "-" and conf >= 0.98 and _looks_like_minus(features):
            strong_ops += 1
        elif char == "/" and conf >= 0.8 and _looks_like_slash(features):
            strong_ops += 1
        elif char in {"1", "7"} and _looks_like_strong_slash_override(features):
            strong_ops += 1
        elif char in {"1", "7"} and conf < 0.995 and _looks_like_slash_override(features):
            strong_ops += 1

    digitish = sum(
        item["char"].isdigit() or item["char"] in {"(", ")", "/"}
        for item in items
    )
    return strong_ops == 0 and digitish >= len(items) - 1


def _map_numeric_char(item):
    raw_char = item["char"]
    features = item["features"]
    looks_like_one = _looks_like_one(features)
    looks_like_open_boundary = _looks_like_open_boundary(features)
    looks_like_close_boundary = _looks_like_close_boundary(features)

    if raw_char in {"/", "(", ")"} and looks_like_one and not (looks_like_open_boundary or looks_like_close_boundary):
        return "1"

    if raw_char == "-" and _looks_like_sparse_seven_override(features):
        return "7"

    if (
        raw_char == "7"
        and features["holes"] == 0
        and features["aspect"] <= 0.55
        and features["right"] >= features["left"] * 1.80
        and features["center"] >= 0.35
        and features["center_row"] >= 0.28
        and features["center_col"] <= 0.35
    ):
        return "1"

    if (
        raw_char == "1"
        and features["holes"] == 0
        and features["aspect"] <= 0.55
        and features["left"] >= features["right"] * 2.20
        and features["center"] <= 0.10
        and features["center_col"] <= 0.30
        and features["bottom"] >= features["top"] * 0.90
    ):
        return "6"

    if raw_char in {"7", "2"} and _looks_like_three_override(features):
        return "3"

    if raw_char == "6" and features["left"] >= features["right"] * 1.08:
        return "6"

    if raw_char == "9" and features["right"] >= features["left"] * 1.08:
        return "9"

    if features["holes"] >= 2:
        return "8"

    if (
        raw_char in {"7", "0", ")"}
        and features["holes"] == 1
        and features["center"] >= 0.18
        and features["right"] >= features["left"] * 1.2
        and features["top"] >= features["bottom"] * 0.95
    ):
        return "9"

    if (
        raw_char == "/"
        and features["holes"] == 0
        and features["aspect"] <= 0.62
        and features["content_w"] <= 10
    ):
        return "1"

    if raw_char == "/" and features["aspect"] >= 0.75:
        if features["bottom"] > features["top"] * 1.12:
            return "2"
        return "3"

    if raw_char in {"(", ")", "6"}:
        if features["holes"] >= 2:
            return "8"
        if features["left"] >= features["right"] * 1.15:
            return "6"
        if (
            features["right"] >= features["left"] * 1.15
            and features["center"] >= 0.18
        ):
            return "9"

    return raw_char


def _override_char(item, prev_char=None, next_char=None):
    features = item["features"]
    raw_char = item["char"]
    conf = float(item["conf"])
    looks_like_one = _looks_like_one(features)
    looks_like_open_boundary = _looks_like_open_boundary(features)
    looks_like_close_boundary = _looks_like_close_boundary(features)

    if (
        raw_char in {"4", "*", "+"}
        and _looks_like_star(features)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "*"

    if (
        raw_char == "/"
        and _looks_like_slash(features)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "/"

    if (
        raw_char in {"1", "7", "/"}
        and _looks_like_slash(features)
        and _has_near_top_model_support(item, "/", min_conf=0.25, max_gap=0.08)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "/"

    if (
        raw_char in {"1", "7"}
        and _looks_like_thin_slash_override(features)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "/"

    if (
        raw_char in {"1", "7"}
        and _looks_like_strong_slash_override(features)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "/"

    if (
        raw_char in {"1", "7"}
        and conf < 0.995
        and _looks_like_slash_override(features)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "/"

    if (
        raw_char in {"(", "1"}
        and looks_like_open_boundary
        and not looks_like_one
        and _has_model_support(item, "(", min_conf=0.07, allow_when_missing=True)
        and _is_open_paren_context(prev_char, next_char)
    ):
        return "("

    if (
        raw_char in {")", "1", "7"}
        and looks_like_close_boundary
        and not looks_like_one
        and _has_model_support(item, ")", min_conf=0.08, allow_when_missing=True)
        and _is_close_paren_context(prev_char, next_char)
    ):
        return ")"

    return raw_char


def _should_merge_as_star(left_item, right_item, prev_char, next_char, median_height):
    if prev_char not in TERM_END_CHARS or next_char not in TERM_START_CHARS:
        return False

    if {left_item["char"], right_item["char"]} - {"(", ")", "1", "/", "4", "7", "+", "*"}:
        return False

    lx, ly, lw, lh = left_item["rect"]
    rx, ry, rw, rh = right_item["rect"]
    gap = rx - (lx + lw)
    x_overlap = min(lx + lw, rx + rw) - max(lx, rx)
    y_overlap = min(ly + lh, ry + rh) - max(ly, ry)
    vertical_gap = max(0, max(ly, ry) - min(ly + lh, ry + rh))
    combined_w = (rx + rw) - lx
    combined_h = max(ly + lh, ry + rh) - min(ly, ry)
    combined_aspect = combined_w / max(1.0, float(combined_h))
    center_dx = abs((lx + (lw / 2.0)) - (rx + (rw / 2.0)))

    touching_fragments = (
        gap <= max(6.0, median_height * 0.12)
        and x_overlap >= -max(6.0, median_height * 0.05)
        and y_overlap >= min(lh, rh) * 0.45
        and 0.45 <= combined_aspect <= 1.2
    )

    vertically_split_fragments = (
        center_dx <= max(8.0, median_height * 0.20)
        and vertical_gap <= max(10.0, median_height * 0.22)
        and x_overlap >= min(lw, rw) * 0.28
        and 0.45 <= combined_aspect <= 1.15
    )

    return touching_fragments or vertically_split_fragments


def _merge_items_as_star(left_item, right_item):
    lx, ly, lw, lh = left_item["rect"]
    rx, ry, rw, rh = right_item["rect"]
    merged_rect = (
        min(lx, rx),
        min(ly, ry),
        max(lx + lw, rx + rw) - min(lx, rx),
        max(ly + lh, ry + rh) - min(ly, ry),
    )
    return {
        "char": "*",
        "conf": max(left_item["conf"], right_item["conf"]),
        "raw_char": f"{left_item['char']}{right_item['char']}",
        "raw_conf": max(left_item["conf"], right_item["conf"]),
        "top_k": [],
        "rect": merged_rect,
        "adjusted": True,
    }


def _drop_edge_operator_noise(items, median_area, median_height):
    trimmed = list(items)

    while len(trimmed) >= 2:
        current = trimmed[0]
        nxt = trimmed[1]
        gap = nxt["rect"][0] - (current["rect"][0] + current["rect"][2])
        area = current["features"]["rect_area"]
        if (
            current["char"] in OPERATORS
            and _looks_like_minus(current["features"])
            and area < max(45.0, median_area * 0.25)
            and gap > median_height * 0.45
        ):
            trimmed.pop(0)
            continue
        break

    while len(trimmed) >= 2:
        current = trimmed[-1]
        prev = trimmed[-2]
        gap = current["rect"][0] - (prev["rect"][0] + prev["rect"][2])
        area = current["features"]["rect_area"]
        if (
            current["char"] in OPERATORS
            and _looks_like_minus(current["features"])
            and area < max(45.0, median_area * 0.25)
            and gap > median_height * 0.45
        ):
            trimmed.pop()
            continue
        break

    return trimmed


def _candidate_char_scores(item):
    current_char = item["char"]
    conf = float(item["conf"])
    features = item.get("features")
    candidates = {
        current_char: conf + 0.55,
    }

    if not features:
        return candidates

    def _add(char, score):
        existing = candidates.get(char)
        if existing is None or score > existing:
            candidates[char] = score

    looks_like_one = _looks_like_one(features)
    looks_like_open_boundary = _looks_like_open_boundary(features)
    looks_like_close_boundary = _looks_like_close_boundary(features)
    slash_top_k = _top_k_score(item, "/")
    slash_near_top = _has_near_top_model_support(
        item,
        "/",
        min_conf=0.25,
        max_gap=0.08,
        allow_when_missing=False,
    )

    if _looks_like_minus(features):
        _add("-", 0.84)

    if _looks_like_star(features):
        _add("*", 0.90 if current_char in {"*", "+", "4"} else 0.86)

    if _looks_like_strong_slash_override(features):
        slash_score = 0.98 if current_char in {"1", "7", "/", ")", "("} else 0.92
        if slash_near_top:
            slash_score = max(slash_score, 1.02 if current_char in {"1", "7", "/"} else 0.94)
        _add("/", slash_score)
    elif _looks_like_slash_override(features):
        slash_score = 0.92 if current_char in {"1", "7", "/", ")", "("} else 0.86
        if slash_near_top:
            slash_score = max(slash_score, 0.98 if current_char in {"1", "7", "/"} else 0.90)
        _add("/", slash_score)
    elif _looks_like_slash(features):
        slash_score = 0.80
        if slash_near_top:
            slash_score = 1.00 if current_char in {"1", "7", "/"} else 0.90
        elif slash_top_k >= 0.10:
            slash_score = max(slash_score, 0.72 + slash_top_k)
        _add("/", slash_score)

    if (
        looks_like_open_boundary
        and not looks_like_one
        and _has_model_support(item, "(", min_conf=0.07, allow_when_missing=True)
    ):
        _add("(", 0.83 if current_char in {"(", "1", "6"} else 0.78)

    if (
        looks_like_close_boundary
        and not looks_like_one
        and _has_model_support(item, ")", min_conf=0.08, allow_when_missing=True)
    ):
        _add(")", 0.83 if current_char in {")", "1", "7"} else 0.78)

    if looks_like_one and not (looks_like_open_boundary or looks_like_close_boundary):
        _add("1", 0.87 if current_char in {"1", "/", "(", ")"} else 0.80)

    return candidates


def _advance_expression_state(state, char):
    balance, expecting_term, prev_char = state

    if expecting_term:
        if char.isdigit():
            return balance, False, char
        if char == "(":
            return balance + 1, True, char
        if char == "-" and (prev_char is None or prev_char in OPERATORS or prev_char == "("):
            return balance, True, char
        return None

    if char.isdigit():
        return balance, False, char
    if char == "(":
        return balance + 1, True, char
    if char in OPERATORS:
        return balance, True, char
    if char == ")":
        if balance <= 0:
            return None
        return balance - 1, False, char
    return None


def _transition_bonus(prev_char, char):
    if prev_char is None:
        return 0.0

    if prev_char in TERM_END_CHARS and char in OPERATORS:
        return 0.05
    if prev_char in OPERATORS and char.isdigit():
        return 0.04
    if prev_char == "(" and (char.isdigit() or char == "-"):
        return 0.04
    if prev_char == ")" and (char.isdigit() or char == "("):
        return 0.03
    if prev_char.isdigit() and char == "(":
        return 0.03
    if prev_char.isdigit() and char.isdigit():
        return 0.01
    return 0.0


def _is_complete_expression_state(state):
    balance, expecting_term, prev_char = state
    if balance != 0 or expecting_term:
        return False
    return prev_char is not None and (prev_char.isdigit() or prev_char == ")")


def _select_expression_consistent_items(items, beam_width=64):
    if len(items) <= 1:
        return items

    initial_state = (0, True, None)
    beams = {
        initial_state: {
            "score": 0.0,
            "items": [],
        }
    }

    for item in items:
        candidate_scores = _candidate_char_scores(item)
        next_beams = {}

        for state, payload in beams.items():
            prev_char = state[2]
            for candidate_char, candidate_score in candidate_scores.items():
                next_state = _advance_expression_state(state, candidate_char)
                if next_state is None:
                    continue

                updated_item = item
                if candidate_char != item["char"]:
                    updated_item = dict(item)
                    updated_item["char"] = candidate_char
                    updated_item["adjusted"] = True

                new_score = (
                    float(payload["score"])
                    + float(candidate_score)
                    + _transition_bonus(prev_char, candidate_char)
                )
                existing = next_beams.get(next_state)
                if existing is None or new_score > existing["score"]:
                    next_beams[next_state] = {
                        "score": new_score,
                        "items": payload["items"] + [updated_item],
                    }

        if not next_beams:
            return items

        ranked = sorted(
            next_beams.items(),
            key=lambda entry: entry[1]["score"],
            reverse=True,
        )[:beam_width]
        beams = {state: payload for state, payload in ranked}

    valid_paths = [
        payload for state, payload in beams.items()
        if _is_complete_expression_state(state)
    ]
    if not valid_paths:
        return items

    best_path = max(valid_paths, key=lambda payload: payload["score"])

    baseline_state = initial_state
    baseline_score = 0.0
    baseline_valid = True
    prev_char = None
    for item in items:
        baseline_state = _advance_expression_state(baseline_state, item["char"])
        if baseline_state is None:
            baseline_valid = False
            break
        baseline_score += float(item["conf"]) + 0.55 + _transition_bonus(prev_char, item["char"])
        prev_char = item["char"]

    if baseline_valid:
        baseline_valid = _is_complete_expression_state(baseline_state)

    if baseline_valid and best_path["score"] < baseline_score + 0.20:
        return items

    return best_path["items"]


def _score_refined_line(refined_items, source_items, best_group_area):
    if not refined_items:
        return float("-inf")

    chars = [item["char"] for item in refined_items]
    digit_count = sum(char.isdigit() for char in chars)
    operator_count = sum(char in OPERATORS for char in chars)
    paren_count = sum(char in {"(", ")"} for char in chars)
    mean_conf = float(np.mean([item["conf"] for item in refined_items]))
    group_area = float(sum(item["area"] for item in source_items))

    min_x = min(item["rect"][0] for item in refined_items)
    min_y = min(item["rect"][1] for item in refined_items)
    max_x = max(item["rect"][0] + item["rect"][2] for item in refined_items)
    max_y = max(item["rect"][1] + item["rect"][3] for item in refined_items)
    span_w = max_x - min_x
    span_h = max_y - min_y
    aspect = span_w / max(1.0, float(span_h))

    score = 0.0
    score += digit_count * 5.0
    score += paren_count * 2.5
    score += operator_count * 1.5
    score += len(refined_items) * 1.4
    score += mean_conf * 2.5
    score += min(3.0, (group_area / max(1.0, best_group_area)) * 3.0)

    if len(refined_items) == 1 and chars[0] in OPERATORS:
        score -= 10.0
    if len(refined_items) <= 2 and digit_count == 0:
        score -= 8.0
    if len(refined_items) <= 2 and operator_count >= 1 and (digit_count + paren_count) <= 1:
        score -= 5.0
    if aspect >= 8.0 and operator_count >= 1 and (digit_count + paren_count) <= 1:
        score -= 8.0
    if digit_count == 0 and operator_count > 0 and paren_count == 0:
        score -= 4.0

    return score


def _normalize_raw_prediction(raw_pred):
    if isinstance(raw_pred, dict):
        normalized = dict(raw_pred)
        normalized["char"] = str(normalized["char"])
        normalized["conf"] = float(normalized["conf"])
        normalized["raw_char"] = str(normalized.get("raw_char", normalized["char"]))
        normalized["raw_conf"] = float(normalized.get("raw_conf", normalized["conf"]))
        return normalized

    char, conf = raw_pred
    conf = float(conf)
    return {
        "char": str(char),
        "conf": conf,
        "raw_char": str(char),
        "raw_conf": conf,
    }


def _refine_line_predictions(
    line_rects,
    line_rois,
    line_raw_predictions,
    apply_context_correction=True,
):
    prepared_predictions = [
        _normalize_raw_prediction(raw_pred)
        for raw_pred in line_raw_predictions
    ]

    if apply_context_correction:
        from segmentation.context_corrector import correct_sequence

        prepared_predictions = correct_sequence(prepared_predictions, line_rois)

    return refine_predictions(line_rects, line_rois, prepared_predictions)


def refine_predictions(rects, roi_images, raw_predictions):
    if not rects or not roi_images or not raw_predictions:
        return []

    items = []
    for rect, roi, raw_pred in zip(rects, roi_images, raw_predictions):
        normalized_raw_pred = _normalize_raw_prediction(raw_pred)
        char = normalized_raw_pred["char"]
        conf = normalized_raw_pred["conf"]

        items.append({
            "char": char,
            "conf": conf,
            "raw_char": normalized_raw_pred["raw_char"],
            "raw_conf": normalized_raw_pred["raw_conf"],
            "top_k": normalized_raw_pred.get("top_k", []),
            "rect": tuple(int(v) for v in rect),
            "roi": roi,
            "features": _extract_features(roi, rect),
            "adjusted": False,
        })

    median_area = float(np.median([item["features"]["rect_area"] for item in items]))
    median_height = float(np.median([item["features"]["rect_h"] for item in items]))

    items = _filter_main_line(items, median_area, median_height)
    if not items:
        return []

    median_area = float(np.median([item["features"]["rect_area"] for item in items]))
    median_height = float(np.median([item["features"]["rect_h"] for item in items]))

    items = [item for item in items if not _should_drop_noise(item, median_area, median_height)]
    if not items:
        return []

    items = _drop_numeric_edge_noise(items, median_area, median_height)
    if not items:
        return []

    median_area = float(np.median([item["features"]["rect_area"] for item in items]))
    median_height = float(np.median([item["features"]["rect_h"] for item in items]))

    numeric_mode = _looks_like_numeric_line(items)

    if numeric_mode:
        merged = _merge_numeric_split_pairs(items, median_height)
    else:
        merged = []
        i = 0
        while i < len(items):
            current = items[i]
            if i + 1 < len(items):
                prev_char = items[i - 1]["char"] if i > 0 else None
                next_char = items[i + 2]["char"] if i + 2 < len(items) else None
                if _should_merge_as_star(current, items[i + 1], prev_char, next_char, median_height):
                    merged.append(_merge_items_as_star(current, items[i + 1]))
                    i += 2
                    continue
            merged.append(current)
            i += 1

    final_items = []
    if numeric_mode:
        for item in merged:
            refined_char = _map_numeric_char(item)
            if refined_char != item["char"]:
                item = dict(item)
                item["char"] = refined_char
                item["adjusted"] = True
            final_items.append(item)
    else:
        for idx, item in enumerate(merged):
            prev_char = final_items[-1]["char"] if final_items else None
            next_char = merged[idx + 1]["char"] if idx + 1 < len(merged) else None
            if "features" in item:
                refined_char = _override_char(item, prev_char, next_char)
                if refined_char != item["char"]:
                    item = dict(item)
                    item["char"] = refined_char
                    item["adjusted"] = True
            final_items.append(item)

        final_items = _drop_edge_operator_noise(final_items, median_area, median_height)
        final_items = _select_expression_consistent_items(final_items)

    return [{
        "char": item["char"],
        "conf": round(float(item["conf"]), 3),
        "raw_char": item["raw_char"],
        "raw_conf": round(float(item["raw_conf"]), 3),
        "top_k": item.get("top_k", []),
        "rect": item["rect"],
        "adjusted": bool(item["adjusted"]),
    } for item in final_items]


def refine_predictions_by_line(
    rects,
    roi_images,
    raw_predictions,
    apply_context_correction=True,
):
    if not rects or not roi_images or not raw_predictions:
        return []

    entries = []
    for rect, roi, raw_pred in zip(rects, roi_images, raw_predictions):
        normalized_raw_pred = _normalize_raw_prediction(raw_pred)
        conf = normalized_raw_pred["conf"]

        x, y, w, h = (int(v) for v in rect)
        entries.append({
            "rect": (x, y, w, h),
            "roi": roi,
            "raw_pred": normalized_raw_pred,
            "area": float(w * h),
            "y_center": y + (h / 2.0),
            "conf": conf,
        })

    heights = [entry["rect"][3] for entry in entries]
    median_height = float(np.median(heights)) if heights else 0.0
    tolerance = max(60.0, median_height * 1.25)

    groups = []
    for entry in sorted(entries, key=lambda item: item["y_center"]):
        placed = False
        for group in groups:
            if abs(entry["y_center"] - group["center"]) <= tolerance:
                group["items"].append(entry)
                group["center"] = float(np.mean([
                    item["y_center"] for item in group["items"]
                ]))
                placed = True
                break
        if not placed:
            groups.append({
                "center": entry["y_center"],
                "items": [entry],
            })

    if len(groups) == 1:
        refined = _refine_line_predictions(
            rects,
            roi_images,
            raw_predictions,
            apply_context_correction=apply_context_correction,
        )
        if not refined:
            return []
        return [{
            "line_index": 0,
            "rect": (
                min(item["rect"][0] for item in refined),
                min(item["rect"][1] for item in refined),
                max(item["rect"][0] + item["rect"][2] for item in refined) - min(item["rect"][0] for item in refined),
                max(item["rect"][1] + item["rect"][3] for item in refined) - min(item["rect"][1] for item in refined),
            ),
            "characters": refined,
        }]

    max_area = max(sum(item["area"] for item in group["items"]) for group in groups)
    significant_groups = []
    for group in groups:
        group_items = group["items"]
        group_area = sum(item["area"] for item in group_items)
        if group_area >= max_area * 0.08 or len(group_items) >= 3:
            significant_groups.append(group)

    if not significant_groups:
        significant_groups = groups

    line_results = []
    best_group_area = max(sum(item["area"] for item in group["items"]) for group in significant_groups)
    for line_index, group in enumerate(sorted(significant_groups, key=lambda item: min(v["rect"][1] for v in item["items"]))):
        sorted_items = sorted(group["items"], key=lambda item: item["rect"][0])
        line_rects = [item["rect"] for item in sorted_items]
        line_rois = [item["roi"] for item in sorted_items]
        line_raw = [item["raw_pred"] for item in sorted_items]
        refined = _refine_line_predictions(
            line_rects,
            line_rois,
            line_raw,
            apply_context_correction=apply_context_correction,
        )
        if not refined:
            continue

        min_x = min(item["rect"][0] for item in refined)
        min_y = min(item["rect"][1] for item in refined)
        max_x = max(item["rect"][0] + item["rect"][2] for item in refined)
        max_y = max(item["rect"][1] + item["rect"][3] for item in refined)

        line_results.append({
            "line_index": line_index,
            "rect": (min_x, min_y, max_x - min_x, max_y - min_y),
            "characters": refined,
            "score": _score_refined_line(refined, sorted_items, best_group_area),
        })

    if len(line_results) > 1:
        best_score = max(item["score"] for item in line_results)
        score_threshold = max(4.0, best_score * 0.42)
        filtered_results = [
            item for item in line_results
            if item["score"] >= score_threshold
        ]
        if filtered_results:
            line_results = filtered_results

    normalized_results = []
    for line_index, item in enumerate(line_results):
        normalized_results.append({
            "line_index": line_index,
            "rect": item["rect"],
            "characters": item["characters"],
        })

    return normalized_results
