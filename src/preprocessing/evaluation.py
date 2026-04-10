from __future__ import annotations


import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

CURRENT_DIR = Path(__file__).resolve().parent # src/model/evaluation
MODEL_DIR = CURRENT_DIR.parent # src/model
SRC_DIR = MODEL_DIR.parent # src
PROJECT_ROOT = SRC_DIR.parent # project root
if str(SRC_DIR) not in sys.path: # Ensure src is in path for absolute imports
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    average: str
    labels: List[Any]
    confusion_matrix: List[List[int]]
    per_class: List[Dict[str, Any]]
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "average": self.average,
            "labels": self.labels,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
            "total_samples": self.total_samples,
        }


# ================================================================
# GENERIC CLASSIFICATION EVALUATION
# ================================================================

def _to_label_vector(values: Sequence[Any], name: str) -> np.ndarray:
    array = np.asarray(values)

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if array.ndim == 1:
        return array

    if array.ndim == 2:
        if array.shape[1] == 1:
            return array.reshape(-1)
        return np.argmax(array, axis=1)

    raise ValueError(
        f"{name} must be a 1D label vector or a 2D score matrix. Received shape {array.shape}."
    )


def _resolve_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Iterable[Any]],
) -> List[Any]:
    if labels is not None:
        return list(labels)

    merged = np.concatenate([y_true, y_pred])
    return list(np.unique(merged))


def evaluate_results(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Iterable[Any]] = None,
    average: str = "macro",
    zero_division: int = 0,
) -> Dict[str, Any]:
    true_labels = _to_label_vector(y_true, "y_true")
    pred_labels = _to_label_vector(y_pred, "y_pred")

    if len(true_labels) != len(pred_labels):
        raise ValueError(
            "y_true and y_pred must have the same number of samples. "
            f"Received {len(true_labels)} and {len(pred_labels)}."
        )

    label_list = _resolve_labels(true_labels, pred_labels, labels)

    accuracy = float(accuracy_score(true_labels, pred_labels))
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=label_list,
        average=average,
        zero_division=zero_division,
    )

    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=label_list,
        average=None,
        zero_division=zero_division,
    )

    matrix = confusion_matrix(true_labels, pred_labels, labels=label_list)
    per_class = []
    for index, label in enumerate(label_list):
        per_class.append(
            {
                "label": label,
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1_score": float(per_class_f1[index]),
                "support": int(support[index]),
            }
        )

    result = EvaluationResult(
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        average=average,
        labels=label_list,
        confusion_matrix=matrix.astype(int).tolist(),
        per_class=per_class,
        total_samples=int(len(true_labels)),
    )
    return result.to_dict()


def format_evaluation(result: Dict[str, Any], decimals: int = 4) -> str:
    lines = [
        "Evaluation metrics",
        f"- Accuracy : {result['accuracy']:.{decimals}f}",
        f"- Precision: {result['precision']:.{decimals}f} ({result['average']} average)",
        f"- Recall   : {result['recall']:.{decimals}f} ({result['average']} average)",
        f"- F1-score : {result['f1_score']:.{decimals}f} ({result['average']} average)",
        f"- Samples  : {result['total_samples']}",
        "",
        "Per-class metrics",
    ]

    for item in result["per_class"]:
        lines.append(
            "- "
            f"{item['label']}: "
            f"precision={item['precision']:.{decimals}f}, "
            f"recall={item['recall']:.{decimals}f}, "
            f"f1={item['f1_score']:.{decimals}f}, "
            f"support={item['support']}"
        )

    return "\n".join(lines)


def evaluate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Iterable[Any]] = None,
    average: str = "macro",
    zero_division: int = 0,
) -> Dict[str, Any]:
    return evaluate_results(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        average=average,
        zero_division=zero_division,
    )


# ================================================================
# EVALUATION 1 - Runtime Keras model on saved dataset arrays
# ================================================================

RUNTIME_MODEL_PATH = MODEL_DIR / "artifacts" / "models" / "best_expression_model.keras"
RUNTIME_DATASET_DIR = MODEL_DIR / "artifacts" / "dataset"
RUNTIME_SUMMARY_PATH = MODEL_DIR / "artifacts" / "models" / "training_summary.json"
EVAL_ARTIFACTS_DIR = PROJECT_ROOT / "eval_artifacts" # This can stay at project root


def _load_runtime_class_names() -> List[str]:
    if RUNTIME_SUMMARY_PATH.exists():
        try:
            payload = json.loads(RUNTIME_SUMMARY_PATH.read_text(encoding="utf-8"))
            class_names = payload.get("class_names")
            if isinstance(class_names, list) and class_names:
                return [str(name) for name in class_names]
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    metadata_path = RUNTIME_DATASET_DIR / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            class_names = payload.get("class_names")
            if isinstance(class_names, list) and class_names:
                return [str(name) for name in class_names]
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return [str(index) for index in range(16)]


def _save_confusion_matrix_figure(
    matrix: np.ndarray,
    label_names: Sequence[str],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = max(8, min(16, len(label_names) * 0.75))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(matrix[row_index, col_index])
            ax.text(
                col_index,
                row_index,
                value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_keras_model(
    model_path: str = RUNTIME_MODEL_PATH,
    x_test_path: str = f"{RUNTIME_DATASET_DIR}/x_val.npy",
    y_test_path: str = f"{RUNTIME_DATASET_DIR}/y_val.npy",
):
    """
    Evaluate the runtime Keras model on saved validation/test arrays.
    Prints loss, accuracy, and a confusion matrix for the available classes.
    """
    import keras
    from sklearn.metrics import classification_report

    print("=" * 60)
    print("EVALUATION 1 - Runtime Keras Model")
    print("=" * 60)

    model_path = Path(model_path)
    x_test_path = Path(x_test_path)
    y_test_path = Path(y_test_path)

    if not model_path.exists():
        print(f"[ERROR] Could not find model at {model_path}")
        return None

    if not x_test_path.exists() or not y_test_path.exists():
        print("[ERROR] Could not find saved dataset arrays for evaluation.")
        return None

    class_names = _load_runtime_class_names()
    model = keras.models.load_model(model_path)
    x_test = np.load(x_test_path).astype("float32") / 255.0
    y_test = np.load(y_test_path)
    x_test = np.expand_dims(x_test, -1)
    output_classes = int(model.output_shape[-1])
    y_test_cat = keras.utils.to_categorical(y_test, output_classes)

    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\n  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    label_names = [class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label) for label in labels]
    metric_payload = evaluate_results(y_test, y_pred, labels=labels)

    print("\n  Summary Metrics:")
    print(f"    Accuracy : {metric_payload['accuracy']:.4f}")
    print(f"    Precision: {metric_payload['precision']:.4f}")
    print(f"    Recall   : {metric_payload['recall']:.4f}")
    print(f"    F1-score : {metric_payload['f1_score']:.4f}")

    print("\n  Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_figure_path = _save_confusion_matrix_figure(
        cm,
        label_names,
        EVAL_ARTIFACTS_DIR / "runtime_confusion_matrix.png",
    )
    print(f"  Labels: {label_names}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Confusion Matrix Figure: {cm_figure_path}")
    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "metrics": metric_payload,
        "labels": label_names,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_figure": str(cm_figure_path),
    }


# ================================================================
# EVALUATION 2 — PyTorch / character model on output_digit
# ================================================================

def evaluate_pytorch_model(
    image_folder: str = "src/segmentation/output_digit",
    expected_chars: Optional[Sequence[str]] = None,
):
    """ 
    Evaluate character model on ROIs in output_digit/.
    expected_chars: list of correct characters in order, e.g., ['3','+','5'].
    If None -> print results only, do not calculate accuracy.
    """
    from segmentation.operator_classifier import predict_character

    print("=" * 60)
    print("EVALUATION 2 — Character Model on output_digit")
    print("=" * 60)

    if not os.path.exists(image_folder):
        print(f"[ERROR] Không tìm thấy folder {image_folder}")
        print("[INFO] Chạy pipeline với 1 ảnh trước để tạo output_digit/")
        return None

    image_list = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    if not image_list:
        print("[ERROR] Không có ảnh nào trong folder!")
        return None 

    print(f"\n  {'File':<20} | {'Predicted':<10} | {'Confidence':<12} | {'Expected':<10} | Result")
    print("  " + "-" * 72)

    correct = 0
    for i, filename in enumerate(image_list):
        img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        char, conf = predict_character(img, normalized=True)
        expected = expected_chars[i] if (expected_chars and i < len(expected_chars)) else "?"
        is_correct = "✅" if expected != "?" and char == expected else ("❌" if expected != "?" else "-")
        if expected != "?" and char == expected:
            correct += 1

        print(f"  {filename:<20} | {char:<10} | {conf * 100:>8.2f}%   | {expected:<10} | {is_correct}")

    if expected_chars:
        total = len(image_list)
        acc = (correct / total * 100.0) if total else 0.0
        print(f"\n  Accuracy: {correct}/{total} = {acc:.1f}%")
        return acc

    return None


# ================================================================
# SEGMENTATION DEBUG HELPERS
# ================================================================

def _default_segment_fn(image_path: str, input_mode: str = "upload"):
    from segmentation.segmentation import segment_image # segmentation is still in src/segmentation

    return segment_image(image_path, debug=False, input_mode=input_mode)


def _default_predict_fn(roi: np.ndarray, normalized: bool = True) -> Tuple[str, float]:
    from model.operator_classifier import predict_character # operator_classifier moved to src/model

    return predict_character(roi, normalized=normalized)


def _safe_read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def segmentation_length_status(true_len: int, pred_len: int) -> Tuple[str, Optional[str]]:
    if pred_len == true_len:
        return "OK_length", None
    if pred_len < true_len:
        return "UNDER_SEGMENT", "Missing characters (characters touching or strokes lost)"
    return "OVER_SEGMENT", "Extra characters (noise, broken box, or over-splitting)"


def evaluate_segmentation_debug(
    image_path: str,
    true_expr: str,
    segment_fn: Optional[Callable[..., Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray, np.ndarray]]] = None,
    input_mode: str = "upload",
) -> Dict[str, Any]:
    """
    Measure basic segmentation errors:
    - correct number of boxes
    - how many boxes rects produces
    - status is OK / UNDER / OVER
    """
    if segment_fn is None:
        segment_fn = _default_segment_fn

    roi_images, rects, thresh, _ = segment_fn(image_path, input_mode=input_mode)
    pred_len = len(roi_images)
    true_len = len(true_expr)
    status, issue = segmentation_length_status(true_len, pred_len)

    rect_list = [tuple(int(v) for v in rect) for rect in rects]
    rect_heights = [r[3] for r in rect_list]
    rect_widths = [r[2] for r in rect_list]

    return {
        "image": image_path,
        "true_expr": true_expr,
        "true_length": true_len,
        "pred_length": pred_len,
        "length_diff": pred_len - true_len,
        "status": status,
        "issue": issue,
        "rects": rect_list,
        "mean_rect_width": float(np.mean(rect_widths)) if rect_widths else 0.0,
        "mean_rect_height": float(np.mean(rect_heights)) if rect_heights else 0.0,
        "has_threshold": thresh is not None,
    }


def visualize_segmentation(
    image_path: str,
    rects: Sequence[Tuple[int, int, int, int]],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Draw bounding boxes + indices to see what segmentation is cutting.
    """
    img = _safe_read_image(image_path)
    canvas = img.copy()

    for i, (x, y, w, h) in enumerate(rects):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            str(i),
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_path, canvas)

    return canvas


def dump_segmentation_artifacts(
    image_path: str,
    true_expr: str,
    out_dir: str = "segmentation_debug",
    segment_fn: Optional[Callable[..., Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray, np.ndarray]]] = None,
    input_mode: str = "upload",
) -> Dict[str, Any]:
    """
    Dump debug images:
    - annotated boxes
    - threshold image
    - each ROI
    - json-like summary dict
    """
    if segment_fn is None:
        segment_fn = _default_segment_fn

    roi_images, rects, thresh, _ = segment_fn(image_path, input_mode=input_mode)
    summary = evaluate_segmentation_debug(
        image_path=image_path,
        true_expr=true_expr,
        segment_fn=segment_fn,
        input_mode=input_mode,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    annotated_path = out_path / "annotated.png"
    thresh_path = out_path / "threshold.png"
    roi_dir = out_path / "rois"
    roi_dir.mkdir(parents=True, exist_ok=True)

    visualize_segmentation(image_path, rects, str(annotated_path))
    if thresh is not None:
        cv2.imwrite(str(thresh_path), thresh)

    for idx, roi in enumerate(roi_images):
        cv2.imwrite(str(roi_dir / f"roi_{idx:02d}.png"), roi)

    summary.update(
        {
            "annotated_path": str(annotated_path),
            "threshold_path": str(thresh_path) if thresh is not None else None,
            "roi_dir": str(roi_dir),
        }
    )
    return summary


def _build_line_predictions(roi_images, rects, predict_fn=None):
    from segmentation.context_corrector import build_raw_predictions # context_corrector is still in src/segmentation
    from segmentation.prediction_refiner import refine_predictions_by_line # prediction_refiner is still in src/segmentation

    raw_predictions = build_raw_predictions(roi_images, predict_fn=predict_fn)
    return refine_predictions_by_line(rects, roi_images, raw_predictions)


def _evaluate_line_predictions(line_predictions):
    from segmentation.expression_parser import build_and_evaluate # expression_parser is still in src/segmentation

    if not line_predictions:
        return "", None, "No valid expression lines were produced."

    if len(line_predictions) == 1:
        raw_chars = [item["char"] for item in line_predictions[0]["characters"]]
        return build_and_evaluate(raw_chars)

    success_count = 0
    for line in line_predictions:
        raw_chars = [item["char"] for item in line["characters"]]
        _, _, line_error = build_and_evaluate(raw_chars)
        if not line_error:
            success_count += 1

    return (
        f"Detected {len(line_predictions)} lines",
        f"{success_count}/{len(line_predictions)} lines evaluated successfully",
        None,
    )


# ================================================================
# EVALUATION 3 — Full pipeline (image -> expression -> result)
# ================================================================

def evaluate_pipeline(test_cases: Sequence[Dict[str, Any]]):
    """
    Test entire pipeline with image list and expected results.

    test_cases = [
        {"image": "input_image/test1.jpg", "expected_expr": "3+5", "expected_result": "8"},
    ]
    """ 
    from segmentation.segmentation import segment_image

    print("=" * 60)
    print("EVALUATION 3 — Full Pipeline")
    print("=" * 60)

    correct_expr = 0
    correct_result = 0
    total = len(test_cases)

    for tc in test_cases:
        image_path = tc["image"]
        expected_expr = tc.get("expected_expr", "?")
        expected_res = tc.get("expected_result", "?")

        roi_images, rects, _, _ = segment_image(image_path, debug=False, input_mode="upload")
        if not roi_images:
            print(f"  [{image_path}] ❌ No characters detected")
            continue

        line_predictions = _build_line_predictions(roi_images, rects)
        expr_str, result_str, error = _evaluate_line_predictions(line_predictions)

        expr_ok = "✅" if expr_str == expected_expr else "❌"
        result_ok = "✅" if result_str == expected_res else "❌"
        if expr_str == expected_expr:
            correct_expr += 1
        if result_str == expected_res:
            correct_result += 1

        seg_debug = evaluate_segmentation_debug(
            image_path=image_path,
            true_expr=expected_expr if expected_expr != "?" else expr_str,
            segment_fn=segment_image,
            input_mode="upload",
        )

        print(f"\n  Image  : {image_path}")
        print(f"  Rects  : {len(rects)} boxes | SegStatus={seg_debug['status']}")
        print(f"  Got    : expr={expr_str!r}  result={result_str!r}")
        print(f"  Expect : expr={expected_expr!r}  result={expected_res!r}")
        print(f"  Status : expr={expr_ok}  result={result_ok}")
        if seg_debug["issue"]:
            print(f"  SegErr : {seg_debug['issue']}")
        if error:
            print(f"  Error  : {error}")

    if total:
        print(f"\n  Expression Accuracy : {correct_expr}/{total} = {correct_expr / total * 100:.1f}%")
        print(f"  Result Accuracy     : {correct_result}/{total} = {correct_result / total * 100:.1f}%")


def evaluate_full_pipeline_debug(
    image_path: str,
    true_expr: str,
    segment_fn: Optional[Callable[..., Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray, np.ndarray]]] = None,
    predict_fn: Optional[Callable[[np.ndarray], Tuple[str, float]]] = None,
    input_mode: str = "upload",
) -> Dict[str, Any]:
    """
    Run a single image and return a full debug dict:
    - segmentation status
    - predicted chars
    - predicted expr
    """
    if segment_fn is None:
        segment_fn = _default_segment_fn
    if predict_fn is None:
        predict_fn = _default_predict_fn

    roi_images, rects, thresh, _ = segment_fn(image_path, input_mode=input_mode)
    line_predictions = _build_line_predictions(roi_images, rects, predict_fn=predict_fn)
    pred_chars = [
        item["char"]
        for line in line_predictions
        for item in line["characters"]
    ]
    pred_confidences = [
        float(item["conf"])
        for line in line_predictions
        for item in line["characters"]
    ]
    pred_expr, pred_result, pred_error = _evaluate_line_predictions(line_predictions)

    seg_debug = evaluate_segmentation_debug(
        image_path=image_path,
        true_expr=true_expr,
        segment_fn=segment_fn,
        input_mode=input_mode,
    )

    return {
        "image": image_path,
        "true_expr": true_expr,
        "pred_expr": pred_expr,
        "pred_result": pred_result,
        "pred_error": pred_error,
        "pred_lines": [
            [item["char"] for item in line["characters"]]
            for line in line_predictions
        ],
        "pred_chars": pred_chars,
        "pred_confidences": pred_confidences,
        "rects": [tuple(int(v) for v in rect) for rect in rects],
        "segmentation": seg_debug,
        "num_rois": len(roi_images),
        "has_threshold": thresh is not None,
    }


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation helpers for the runtime model and pipeline.")
    parser.add_argument(
        "--runtime-model",
        action="store_true",
        help="Print runtime model metrics and confusion matrix using x_val/y_val.",
    )
    parser.add_argument(
        "--sample-demo",
        action="store_true",
        help="Run the small hardcoded demo metrics example instead of the real runtime model evaluation.",
    )
    args = parser.parse_args()

    if args.sample_demo:
        sample_true = [0, 1, 2, 2, 1, 0, 3, 3]
        sample_pred = [0, 1, 2, 1, 1, 0, 3, 2]
        metrics = evaluate_results(sample_true, sample_pred)
        print(format_evaluation(metrics))
    else:
        evaluate_keras_model()
