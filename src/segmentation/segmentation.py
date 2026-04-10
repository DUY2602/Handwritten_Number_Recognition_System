"""
Refactored segmentation entrypoint.

The public API remains `segment_image(...)`, while the implementation is split
across smaller modules by processing step.
"""

import os
import sys

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.preprocessing import normalize_binary_character

from .candidate_ops import (
    _extract_canvas_rects,
    _extend_region_along_row,
    _segment_generic_region,
    _segment_guided_handwriting_row,
    _segment_notebook_band_fallback,
    _select_best_region,
    _select_best_upload_candidate,
    _validate_band_result,
    _validate_generic_result,
)
from .image_ops import (
    _binarize,
    _binarize_canvas_strokes,
    _binarize_dark_strokes,
    _enhance,
    _remove_grid_lines,
    _score_threshold,
    _to_ink_map,
)
from .io_utils import DEBUG_ROOT, OUTPUT_ROOT, _clean_dir, _mode_artifact_dir, read_image_safe
from .logging_utils import get_logger, log_info_print
from .rect_ops import (
    _crop_with_padding,
    _extract_rects_from_thresh,
    _filter_rects,
    _merge_broken_parts,
    _merge_single_digit_pairs,
    _operator_preserving_filter,
    _remove_inside_boxes,
    _sort_by_row_col,
    _split_wide_rects,
)

logger = get_logger(__name__)
print = lambda *args, **kwargs: log_info_print(*args, logger=logger, **kwargs)


def segment_image(image_path, debug=False, input_mode="upload"):
    """
    Extract characters from a mobile phone image.

    Upload mode:
        1. Ink-map + enhance
        2. Try NOTEBOOK BAND-FIRST segmentation first
        3. If invalid, fallback to guided/generic region segmentation
        4. If still invalid, fallback to full-image split/merge pipeline

    Draw mode:
        - Keep the original canvas contour path

    Trả về:
        roi_images  : list[np.ndarray]  - normalized 28x28 images
        valid_rects : list[tuple]       - corresponding (x, y, w, h) boxes
        thresh      : np.ndarray        - processed binary image
        img_display : np.ndarray        - original image with bounding boxes
    """
    input_mode = str(input_mode or "upload").strip().lower()
    if input_mode not in {"upload", "draw"}:
        input_mode = "upload"

    print(f"\n[INFO] Xu ly: {image_path}")
    output_dir = _mode_artifact_dir(OUTPUT_ROOT, input_mode)
    debug_dir = _mode_artifact_dir(DEBUG_ROOT, input_mode)

    _clean_dir(output_dir)
    if debug:
        _clean_dir(debug_dir)

    img = read_image_safe(image_path)
    if img is None:
        print("[ERROR] Could not read image.")
        return [], [], None, None

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_display = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h_img, w_img = img.shape[:2]
    print(f"[INFO] Image dimensions: {w_img}x{h_img}")
    print(f"[INFO] Input mode: {input_mode}")

    filtered = []
    thresh = None

    if input_mode == "upload":
        ink = _to_ink_map(img)
        enhanced = _enhance(ink)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        if debug:
            cv2.imwrite(os.path.join(debug_dir, "01_enhanced.png"), enhanced)

        band_result = _segment_notebook_band_fallback(enhanced, h_img, w_img)
        if band_result is not None:
            band_ok, band_reason = _validate_band_result(band_result, w_img, h_img)
        else:
            band_ok, band_reason = False, "No band result"

        if band_ok:
            print(f"\n[SEGMENT] BAND-FIRST segmentation: ACCEPTED ({band_reason})")
            thresh = band_result["thresh"]
            filtered = sorted(band_result["rects"], key=lambda r: r[0])
            
            if filtered:
                filtered = _split_wide_rects(filtered, thresh)
                med_h = float(np.median([r[3] for r in filtered]))
                med_w = float(np.median([r[2] for r in filtered]))
                filtered = _merge_broken_parts(filtered, med_h, med_w)
                print(f"[SEGMENT] Found {len(filtered)} characters in band after splitting")

            if debug:
                cv2.imwrite(os.path.join(debug_dir, "02_band_thresh.png"), thresh)
                line_mask = band_result.get("line_mask")
                if line_mask is not None:
                    cv2.imwrite(os.path.join(debug_dir, "02b_band_line_mask.png"), line_mask)
        else:
            print(f"\n[SEGMENT] BAND-FIRST segmentation: REJECTED ({band_reason})")
            print("[SEGMENT] Evaluating GUIDED and REGION candidates...")

            thresh_candidates = [
                _binarize(enhanced, h_img, w_img),
                _binarize_dark_strokes(gray, h_img, w_img),
            ]
            thresh = max(thresh_candidates, key=_score_threshold)
            if debug:
                cv2.imwrite(os.path.join(debug_dir, "02_thresh_raw.png"), thresh)

            thresh = _remove_grid_lines(thresh, h_img, w_img)
            if debug:
                cv2.imwrite(os.path.join(debug_dir, "03_no_lines.png"), thresh)

            fg_ratio = np.count_nonzero(thresh) / max(1.0, float(thresh.size))
            if fg_ratio < 0.025:
                heal_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, heal_k, iterations=1)
                if debug:
                    cv2.imwrite(os.path.join(debug_dir, "03b_healed_sparse.png"), thresh)

            filtered = []
            region_for_seg = None
            upload_candidates = []
            is_valid = False
            reason = "No upload candidate"

            print("\n[SEGMENT] Attempting GUIDED handwriting-focus segmentation...")
            guided_result = _segment_guided_handwriting_row(img, gray, thresh, w_img, h_img)
            if guided_result is not None:
                print(
                    "[SEGMENT] GUIDED handwriting candidate: READY "
                    f"({guided_result['reason']}) score={guided_result['score']:.2f}"
                )
                upload_candidates.append({
                    "name": "guided",
                    "reason": guided_result["reason"],
                    "rects": sorted(guided_result["rects"], key=lambda r: r[0]),
                    "thresh": guided_result["thresh"],
                    "region": guided_result["region"],
                    "score": guided_result["score"],
                    "color_mask": guided_result.get("color_mask"),
                    "focus_mask": guided_result.get("focus_mask"),
                })
            else:
                print("[SEGMENT] GUIDED handwriting candidate: REJECTED")

            print("[SEGMENT] Attempting GENERIC REGION-FIRST segmentation...")
            best_region, region_score, _ = _select_best_region(thresh, w_img, h_img)
            region_for_seg = best_region

            if best_region is not None:
                print(f"[SEGMENT] Best region score: {region_score:.2f} at {best_region}")
                region_for_seg = _extend_region_along_row(thresh, best_region, w_img, h_img)
                if region_for_seg != best_region:
                    print(f"[SEGMENT] Extended region along row: {region_for_seg}")

                region_rects = _segment_generic_region(thresh, region_for_seg, w_img, h_img)
                region_ok, region_reason = _validate_generic_result(region_rects, thresh, w_img, h_img)
                if region_ok:
                    print(f"[SEGMENT] REGION candidate: READY ({region_reason})")
                    upload_candidates.append({
                        "name": "region",
                        "reason": region_reason,
                        "rects": sorted(region_rects, key=lambda r: r[0]),
                        "thresh": thresh,
                        "region": region_for_seg,
                    })
                else:
                    print(f"[SEGMENT] REGION candidate: REJECTED ({region_reason})")
            else:
                print("[SEGMENT] REGION candidate: REJECTED (No candidate region)")

            chosen_candidate, scored_candidates = _select_best_upload_candidate(upload_candidates, w_img, h_img)
            for candidate in scored_candidates:
                metrics = candidate.get("selection_metrics", {})
                print(
                    "[SEGMENT] Candidate "
                    f"{candidate['name'].upper()}: quality={candidate['selection_score']:.2f} "
                    f"count={metrics.get('count', 0)} span={metrics.get('span_ratio', 0.0):.2f} "
                    f"jitter={metrics.get('row_jitter', 0.0):.2f} "
                    f"slim={metrics.get('slim_ratio', 0.0):.2f} "
                    f"tiny={metrics.get('tiny_ratio', 0.0):.2f}"
                )

            if chosen_candidate is not None:
                thresh = chosen_candidate["thresh"]
                filtered = sorted(chosen_candidate["rects"], key=lambda r: r[0])
                region_for_seg = chosen_candidate.get("region")
                is_valid = True
                reason = chosen_candidate.get("reason", "")
                print(
                    "[SEGMENT] Selected upload candidate: "
                    f"{chosen_candidate['name'].upper()} (quality={chosen_candidate['selection_score']:.2f})"
                )

                if debug and chosen_candidate["name"] == "guided":
                    color_mask = chosen_candidate.get("color_mask")
                    focus_mask = chosen_candidate.get("focus_mask")
                    if color_mask is not None:
                        cv2.imwrite(os.path.join(debug_dir, "04_guided_color_mask.png"), color_mask)
                    if focus_mask is not None:
                        cv2.imwrite(os.path.join(debug_dir, "04b_guided_focus_mask.png"), focus_mask)
                    cv2.imwrite(os.path.join(debug_dir, "04c_guided_thresh.png"), thresh)

            if is_valid:
                print(f"[SEGMENT] Selected segmentation: ACCEPTED ({reason})")
                if filtered:
                    filtered = _split_wide_rects(filtered, thresh)
                    med_h = float(np.median([r[3] for r in filtered]))
                    med_w = float(np.median([r[2] for r in filtered]))
                    filtered = _merge_broken_parts(filtered, med_h, med_w)
                print(f"[SEGMENT] Found {len(filtered)} characters in selected region after splitting")
                if debug and region_for_seg is not None:
                    x0, y0, x1, y1 = region_for_seg
                    region_vis = img_display.copy()
                    cv2.rectangle(region_vis, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(debug_dir, "04_region_selected.png"), region_vis)
            else:
                print(f"[SEGMENT] Candidate selection: REJECTED ({reason})")
                print("[SEGMENT] Falling back to FULL IMAGE segmentation...")

                initial, merged, filtered_small = _extract_rects_from_thresh(thresh, w_img, h_img)
                split_source = merged if merged else initial
                if not split_source:
                    split_source = filtered_small

                filtered = _split_wide_rects(split_source, thresh) if split_source else []
                if filtered:
                    filtered = _filter_rects(filtered, thresh, w_img, h_img)
                    if filtered:
                        med_h = float(np.median([r[3] for r in filtered]))
                        med_w = float(np.median([r[2] for r in filtered]))
                        filtered = _merge_broken_parts(filtered, med_h, med_w)
                        filtered = _merge_single_digit_pairs(filtered, thresh)
                        filtered = _remove_inside_boxes(filtered, thresh, w_img, h_img)
                        filtered = _operator_preserving_filter(filtered, thresh, w_img, h_img)

                if debug:
                    cv2.imwrite(os.path.join(debug_dir, "04_fallback_result.png"), thresh)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        if debug:
            cv2.imwrite(os.path.join(debug_dir, "01_gray.png"), gray)

        thresh = _binarize_canvas_strokes(gray, h_img, w_img)
        if debug:
            cv2.imwrite(os.path.join(debug_dir, "02_thresh_canvas.png"), thresh)

        _, _, filtered = _extract_canvas_rects(thresh, w_img, h_img)

    print(f"[INFO] After segmentation: {len(filtered)} boxes")

    if filtered and thresh is not None:
        filtered = _merge_single_digit_pairs(filtered, thresh)
        filtered = _remove_inside_boxes(filtered, thresh, w_img, h_img)
        
        def _tighten_rect(rect, map_img):
            x, y, w, h = rect
            roi = map_img[max(0, y):y + h, max(0, x):x + w]
            ys, xs = np.where(roi > 0)
            if len(xs) == 0:
                return rect
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            return (x + x0, y + y0, x1 - x0, y1 - y0)
            
        filtered = [_tighten_rect(r, thresh) for r in filtered]

    sorted_rects = _sort_by_row_col(filtered)

    roi_images = []
    valid_rects = []

    for idx, (x, y, w, h) in enumerate(sorted_rects):
        roi = _crop_with_padding(thresh, x, y, w, h)
        if roi is None:
            continue

        final_img = normalize_binary_character(roi)
        out_path = os.path.join(output_dir, f"digit_{idx:03d}.png")
        cv2.imwrite(out_path, final_img)

        roi_images.append(final_img)
        valid_rects.append((x, y, w, h))

        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            img_display,
            str(idx),
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 0),
            1,
        )

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "99_final_result.png"), img_display)

    print(f"[INFO] Tong so duoc tach: {len(roi_images)}")
    if valid_rects:
        print(f"[INFO] Average box height: {np.mean([h for _, _, _, h in valid_rects]):.1f}px")
    else:
        print("[INFO] Average box height: 0.0px")

    return roi_images, valid_rects, thresh, img_display


__all__ = ["segment_image", "_score_threshold", "read_image_safe"]
