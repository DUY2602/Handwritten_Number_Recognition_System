import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from segmentation.context_corrector import build_raw_predictions
from segmentation.expression_parser import build_and_evaluate
from segmentation.prediction_refiner import refine_predictions_by_line
from segmentation.segmentation import segment_image


def run_expression_pipeline(image_path, show_visualization=True, input_mode="upload"):
    """
    Quy trình đầy đủ: phân đoạn → phân loại → phân tích → tính toán → trực quan hóa.
    """

    # --- Bước 1: Phân đoạn hình ảnh ---
    print("\n[STEP 1] Segmenting image...")
    roi_images, rects, thresh, img_display = segment_image(
        image_path,
        input_mode=input_mode,
    )
    
    if not roi_images:
        return "", None, "No characters detected."
    
    print(f"Found {len(roi_images)} character(s).")


    # --- Bước 2: Phân loại từng ký tự ---
    print("\n[STEP 2] Classifying characters...")
    raw_predictions = build_raw_predictions(roi_images)
    for i, item in enumerate(raw_predictions):
        print(f"  ROI {i}: predicted '{item['char']}' (confidence {item['conf']:.3f})")

    line_predictions = refine_predictions_by_line(rects, roi_images, raw_predictions)
    if not line_predictions:
        return "", None, "The detected characters could not be grouped into a valid expression line."

    predictions = [
        item
        for line in line_predictions
        for item in line["characters"]
    ]
    refined_rects = [item["rect"] for item in predictions]
    prediction_pairs = [(item["char"], item["conf"]) for item in predictions]

    # --- Bước 3: Tính toán biểu thức ---
    print("\n[STEP 3] Building and evaluating expression...")
    if len(line_predictions) == 1:
        raw_chars = [item["char"] for item in line_predictions[0]["characters"]]
        expression_str, result_str, error = build_and_evaluate(raw_chars)
    else:
        success_count = 0
        for idx, line in enumerate(line_predictions, start=1):
            raw_chars = [item["char"] for item in line["characters"]]
            line_expr, line_result, line_error = build_and_evaluate(raw_chars)
            if line_error:
                print(f"  Line {idx}: {line_expr} -> ERROR ({line_error})")
            else:
                print(f"  Line {idx}: {line_expr} = {line_result}")
                success_count += 1

        expression_str = f"Detected {len(line_predictions)} lines"
        result_str = f"{success_count}/{len(line_predictions)} lines evaluated successfully"
        error = None
    
    print(f"\n  Expression : {expression_str}")
    if error:
        print(f"  Error      : {error}")
    else:
        print(f"  Result     : {result_str}")
    # --- Bước 4: Trực quan hóa ---
    if show_visualization:
        print("\n[STEP 4] Showing results...")
        _visualize(img_display, refined_rects, prediction_pairs, roi_images, thresh, expression_str, result_str)
    
    print("\n" + "=" * 60)
    return expression_str, result_str, error


def _visualize(img_display, rects, predictions, roi_images, thresh, expr_str, result_str):
    vis = img_display.copy()
    
    # Chú thích hộp 
    for (x, y, w, h), (char, conf) in zip(rects, predictions):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{char} ({conf:.2f})"
        cv2.putText(vis, label, (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Tạo hình
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 2", fontsize=14, fontweight='bold')


    # Bảng 1: Ảnh gốc 
    axes[0, 0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Final Result (Merged Boxes)")
    axes[0, 0].axis('off')

    # Bảng 2: Ngưỡng nhị phân
    axes[0, 1].imshow(thresh, cmap='gray')
    axes[0, 1].set_title("Binary Input")
    axes[0, 1].axis('off')

    # Bảng 3: Dải ROI
    if roi_images:
        strip_h = 28
        strip_w = 28 * len(roi_images) + 4 * (len(roi_images) - 1)
        strip = np.zeros((strip_h, strip_w), dtype=np.uint8)
        for idx, roi in enumerate(roi_images):
            strip[:, idx * 32:idx * 32 + 28] = roi
        axes[1, 0].imshow(strip, cmap='gray')
        axes[1, 0].set_title("Segmented ROIs (left to right)")
        axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, "No ROIs", ha='center', va='center')
        axes[1, 0].axis('off')

    # Bảng 4: Văn bản kết quả
    axes[1, 1].axis('off')
    result_display = result_str if result_str else "ERROR"
    text_block = (
        f"Recognised Expression:\n"
        f"    {expr_str}\n\n"
        f"Computed Result:\n"
        f"    {result_display}"
    )
    axes[1, 1].text(0.1, 0.5, text_block,
                    fontsize=16, fontfamily='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'input_image/test.jpg'
    input_mode = sys.argv[2] if len(sys.argv) > 2 else "upload"
    run_expression_pipeline(path, input_mode=input_mode)
