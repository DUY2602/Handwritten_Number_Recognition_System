import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from segmentation.runtime_pipeline import analyze_expression_image


def run_expression_pipeline(image_path, show_visualization=True, input_mode="upload"):
    """
    Full pipeline: segment -> classify -> parse -> evaluate -> visualize.
    """

    print("\n[STEP 1] Segmenting image...")
    analysis = analyze_expression_image(
        image_path,
        input_mode=input_mode,
    )
    roi_images = analysis["roi_images"]
    thresh = analysis["thresh"]
    img_display = analysis["img_display"]
    raw_predictions = analysis["raw_predictions"]
    line_predictions = analysis["line_predictions"]
    expression_str = analysis["expression"]
    result_str = analysis["result"]
    error = analysis["error"]

    if not roi_images:
        return "", None, analysis["fatal_error"]

    print(f"Found {len(roi_images)} character(s).")

    print("\n[STEP 2] Classifying characters...")
    for i, item in enumerate(raw_predictions):
        print(f"  ROI {i}: predicted '{item['char']}' (confidence {item['conf']:.3f})")

    if not line_predictions:
        return "", None, analysis["fatal_error"]

    predictions = [
        item
        for line in line_predictions
        for item in line["characters"]
    ]
    refined_rects = [item["rect"] for item in predictions]
    prediction_pairs = [(item["char"], item["conf"]) for item in predictions]

    print("\n[STEP 3] Building and evaluating expression...")
    if len(line_predictions) > 1:
        for idx, line in enumerate(analysis["lines"], start=1):
            if line["error"]:
                print(f"  Line {idx}: {line['expression']} -> ERROR ({line['error']})")
            else:
                print(f"  Line {idx}: {line['expression']} = {line['result']}")

    print(f"\n  Expression : {expression_str}")
    if error:
        print(f"  Error      : {error}")
    else:
        print(f"  Result     : {result_str}")

    if show_visualization:
        print("\n[STEP 4] Showing results...")
        _visualize(
            img_display,
            refined_rects,
            prediction_pairs,
            roi_images,
            thresh,
            expression_str,
            result_str,
        )

    print("\n" + "=" * 60)
    return expression_str, result_str, error


def _visualize(img_display, rects, predictions, roi_images, thresh, expr_str, result_str):
    vis = img_display.copy()

    for (x, y, w, h), (char, conf) in zip(rects, predictions):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{char} ({conf:.2f})"
        cv2.putText(
            vis,
            label,
            (x, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 2", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Final Result (Merged Boxes)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(thresh, cmap="gray")
    axes[0, 1].set_title("Binary Input")
    axes[0, 1].axis("off")

    if roi_images:
        strip_h = 28
        strip_w = 28 * len(roi_images) + 4 * (len(roi_images) - 1)
        strip = np.zeros((strip_h, strip_w), dtype=np.uint8)
        for idx, roi in enumerate(roi_images):
            strip[:, idx * 32:idx * 32 + 28] = roi
        axes[1, 0].imshow(strip, cmap="gray")
        axes[1, 0].set_title("Segmented ROIs (left to right)")
        axes[1, 0].axis("off")
    else:
        axes[1, 0].text(0.5, 0.5, "No ROIs", ha="center", va="center")
        axes[1, 0].axis("off")

    axes[1, 1].axis("off")
    result_display = result_str if result_str else "ERROR"
    text_block = (
        f"Recognised Expression:\n"
        f"    {expr_str}\n\n"
        f"Computed Result:\n"
        f"    {result_display}"
    )
    axes[1, 1].text(
        0.1,
        0.5,
        text_block,
        fontsize=16,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "input_image/test.jpg"
    input_mode = sys.argv[2] if len(sys.argv) > 2 else "upload"
    run_expression_pipeline(path, input_mode=input_mode)
