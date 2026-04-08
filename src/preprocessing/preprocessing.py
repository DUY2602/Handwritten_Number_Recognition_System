import cv2
import numpy as np
import torch


TARGET_SIZE = 28
INNER_SIZE = 22


def _to_grayscale(image_input):
    if image_input is None:
        raise ValueError("image_input must not be None")

    if len(image_input.shape) == 3 and image_input.shape[2] == 3:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image_input.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def _looks_like_white_on_black(gray):
    if gray.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
        return False

    edges = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    edge_mean = float(edges.mean())
    bright_ratio = np.count_nonzero(gray >= 180) / float(gray.size)
    dark_ratio = np.count_nonzero(gray <= 40) / float(gray.size)

    return (
        edge_mean <= 35.0
        and bright_ratio >= 0.02
        and bright_ratio <= 0.45
        and dark_ratio >= 0.45
    )


def _to_ink_map(image_input):
    gray = _to_grayscale(image_input)
    if _looks_like_white_on_black(gray):
        return gray

    if len(image_input.shape) == 3:
        if image_input.shape[2] == 4:
            color = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        else:
            color = image_input

        if color.dtype != np.uint8:
            color = np.clip(color, 0, 255).astype(np.uint8)

        gray_darkness = cv2.subtract(255, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
        channel_min = color.min(axis=2).astype(np.uint8)
        color_darkness = cv2.subtract(255, channel_min)
        return cv2.max(gray_darkness, color_darkness)

    return cv2.subtract(255, gray)


def to_ink_map(image_input):
    """
    Backward-compatible public alias for older imports.
    """
    return _to_ink_map(image_input)


def _binary_score(binary):
    foreground = int(np.count_nonzero(binary))
    if foreground == 0:
        return float("-inf")

    coords = cv2.findNonZero(binary)
    if coords is None:
        return float("-inf")

    ratio = foreground / float(binary.size)
    edges = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    edge_ratio = np.count_nonzero(edges) / max(1.0, float(edges.size))
    _, _, w, h = cv2.boundingRect(coords)
    bbox_fill = foreground / max(1.0, float(w * h))

    # A handwritten character should occupy a minority of the canvas and
    # should not flood the image borders. Favor compact foreground to avoid
    # letting sparse edge noise win over a thinner but cleaner stroke.
    return -abs(ratio - 0.12) - (edge_ratio * 0.8) + (min(bbox_fill, 0.6) * 0.18)


def clean_binary_mask(binary):
    binary_u8 = np.where(binary > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num_labels <= 2:
        return binary_u8

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    dominant_label = 1 + int(np.argmax(component_areas))
    dominant_x = int(stats[dominant_label, cv2.CC_STAT_LEFT])
    dominant_y = int(stats[dominant_label, cv2.CC_STAT_TOP])
    dominant_w = int(stats[dominant_label, cv2.CC_STAT_WIDTH])
    dominant_h = int(stats[dominant_label, cv2.CC_STAT_HEIGHT])
    dominant_area = int(stats[dominant_label, cv2.CC_STAT_AREA])

    # Looser cleanup for real-world handwritten digits:
    # keep the dominant component, but also preserve plausible nearby pieces
    # such as broken loops, tails, and pen-lift fragments.
    pad = max(3, int(round(max(dominant_w, dominant_h) * 0.55)))
    x0 = dominant_x - pad
    y0 = dominant_y - pad
    x1 = dominant_x + dominant_w + pad
    y1 = dominant_y + dominant_h + pad
    dominant_cx = dominant_x + (dominant_w / 2.0)
    dominant_cy = dominant_y + (dominant_h / 2.0)

    keep_labels = {dominant_label}
    min_neighbor_area = max(3, int(round(dominant_area * 0.035)))
    min_significant_area = max(6, int(round(dominant_area * 0.20)))

    for label in range(1, num_labels):
        if label == dominant_label:
            continue

        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_neighbor_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]

        overlaps_expanded_box = not (x + w < x0 or x > x1 or y + h < y0 or y > y1)
        centroid_close = (
            abs(float(cx) - dominant_cx) <= max(7.0, dominant_w * 1.15)
            and abs(float(cy) - dominant_cy) <= max(7.0, dominant_h * 1.15)
        )
        touching_or_near = (
            x <= x1 + 2 and x + w >= x0 - 2
            and y <= y1 + 2 and y + h >= y0 - 2
        )
        significant_component = area >= min_significant_area
        slender_support = (
            area >= min_neighbor_area
            and (h >= max(4, int(round(dominant_h * 0.35))) or w >= max(4, int(round(dominant_w * 0.35))))
            and (overlaps_expanded_box or centroid_close)
        )

        if overlaps_expanded_box or centroid_close or touching_or_near or significant_component or slender_support:
            keep_labels.add(label)

    filtered = np.zeros_like(binary_u8)
    for label in keep_labels:
        filtered[labels == label] = 255

    if np.count_nonzero(filtered) == 0:
        return binary_u8

    return filtered


def _stroke_is_thin(binary):
    coords = cv2.findNonZero(binary)
    if coords is None:
        return False

    x, y, w, h = cv2.boundingRect(coords)
    if w <= 0 or h <= 0:
        return False

    area = float(np.count_nonzero(binary))
    bbox_area = float(max(1, w * h))
    fill_ratio = area / bbox_area
    return fill_ratio < 0.20 and min(w, h) >= 6


def _strengthen_strokes(binary):
    # Conservative healing: close tiny breaks first, then dilate slightly only
    # when the glyph is clearly too thin.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    healed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    if _stroke_is_thin(healed):
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        healed = cv2.dilate(healed, kernel_dilate, iterations=1)

    return healed


def remove_small_components(binary, min_component_area):
    """
    Backward-compatible helper for full-image threshold cleanup.
    """
    binary_u8 = np.where(binary > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num_labels <= 1:
        return binary_u8

    filtered = np.zeros_like(binary_u8)
    kept_any = False

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        filtered[labels == label] = 255
        kept_any = True

    if kept_any:
        return filtered

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    filtered[labels == largest_label] = 255
    return filtered


def _binarize_character(image_input):
    ink = _to_ink_map(image_input)
    blurred = cv2.GaussianBlur(ink, (3, 3), 0)

    _, binary_otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -2
    )

    candidates = []
    kernel_open = np.ones((2, 2), dtype=np.uint8)
    for candidate in (binary_otsu, binary_adaptive):
        denoised = cv2.medianBlur(candidate, 3)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_open)
        denoised = clean_binary_mask(denoised)
        candidates.append(denoised)

    chosen = max(candidates, key=_binary_score)
    return chosen


def _center_on_canvas(binary):
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]

    area = float(np.count_nonzero(cropped))
    bbox_area = float(max(1, w * h))
    fill_ratio = area / bbox_area

    # Make thin real-world glyphs occupy more of the canvas than clean,
    # already-bold blobs.
    target_inner = INNER_SIZE
    if fill_ratio < 0.18:
        target_inner = min(TARGET_SIZE - 4, INNER_SIZE + 2)
    if fill_ratio < 0.12:
        target_inner = min(TARGET_SIZE - 3, INNER_SIZE + 4)

    if h >= w:
        new_h = target_inner
        new_w = max(1, int(round(w * target_inner / float(h))))
    else:
        new_w = target_inner
        new_h = max(1, int(round(h * target_inner / float(w))))

    interpolation = cv2.INTER_LINEAR if max(new_w, new_h) > max(w, h) else cv2.INTER_AREA
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    x_offset = (TARGET_SIZE - new_w) // 2
    y_offset = (TARGET_SIZE - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        shift_x = int(round((TARGET_SIZE / 2.0) - center_x))
        shift_y = int(round((TARGET_SIZE / 2.0) - center_y))
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (TARGET_SIZE, TARGET_SIZE))

    return canvas


def normalize_binary_character(binary_input):
    binary = np.where(binary_input > 0, 255, 0).astype(np.uint8)
    binary = clean_binary_mask(binary)
    binary = _strengthen_strokes(binary)
    binary = clean_binary_mask(binary)
    return _center_on_canvas(binary)


def preprocess(image_input):
    """
    Normalize a character image to the same 28x28 white-on-black format used
    by the classifier.
    """
    binary = _binarize_character(image_input)
    return _center_on_canvas(binary)


def preprocess_for_torch(image_input, assume_normalized=False):
    processed = _to_grayscale(image_input) if assume_normalized else preprocess(image_input)
    tensor = torch.tensor(processed, dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


def preprocess_for_keras(image_input, assume_normalized=False):
    processed = _to_grayscale(image_input) if assume_normalized else preprocess(image_input)
    normalized = processed.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))
