import argparse
import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATASET_DIR = CURRENT_DIR / "artifacts" / "dataset"
MODELS_DIR = CURRENT_DIR / "artifacts" / "models"
FEEDBACK_DIR = CURRENT_DIR / "artifacts" / "feedback"
MODEL_PATH = MODELS_DIR / "best_expression_model.keras"
SUMMARY_PATH = MODELS_DIR / "training_summary.json"
LOG_PATH = MODELS_DIR / "training_log.csv"

SEED = 42
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 384
DEFAULT_LEARNING_RATE = 8e-6

TARGET_LABEL_NAMES = ("*", "/")
BOUNDARY_LABEL_NAMES = ("1", "(", ")")
FOCUS_LABEL_NAMES = TARGET_LABEL_NAMES + BOUNDARY_LABEL_NAMES
DEFAULT_TARGET_DUPLICATION = {
    "*": 1,
    "/": 0,
}
DEFAULT_TARGET_BASE_WEIGHTS = {
    "*": 1.6,
    "/": 1.8,
    "1": 1.15,
    "(": 1.1,
    ")": 1.1,
}
TARGET_FEEDBACK_WEIGHT = 2.5
NON_TARGET_FEEDBACK_WEIGHT = 1.25
DEFAULT_FEEDBACK_AUG_REPEATS = {
    "*": 24,
    "/": 24,
}
DEFAULT_HARD_EXAMPLE_LIMITS = {
    "*": 512,
    "/": 2048,
    "1": 1536,
    "(": 1024,
    ")": 1024,
}
DEFAULT_HARD_AUG_REPEATS = {
    "*": 2,
    "/": 2,
    "1": 1,
    "(": 1,
    ")": 1,
}
DEFAULT_HARD_FALLBACK_LIMITS = {
    "*": 256,
}
DEFAULT_HARD_CONFIDENCE_THRESHOLD = 0.992
DEFAULT_HARD_MARGIN_THRESHOLD = 0.040
DEFAULT_HARD_WEIGHT_MULTIPLIER = 1.20
CONFUSION_PARTNERS = {
    "*": {"+", "4"},
    "/": {"1", "(", ")"},
    "1": {"/", "(", ")", "7"},
    "(": {"/", "1"},
    ")": {"/", "1"},
}


@dataclass
class EvalMetrics:
    accuracy: float
    macro_f1: float
    macro_recall: float
    target_macro_f1: float
    target_macro_recall: float
    boundary_macro_f1: float
    boundary_macro_recall: float
    priority_score: float
    per_class: dict


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _load_class_names() -> list[str]:
    if SUMMARY_PATH.exists():
        try:
            payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            class_names = payload.get("class_names")
            if isinstance(class_names, list) and class_names:
                return [str(name) for name in class_names]
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    metadata = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    return [str(name) for name in metadata["class_names"]]


def _normalize_feedback_label(raw_label: str) -> str | None:
    label = raw_label.strip()
    aliases = {
        "plus": "+",
        "minus": "-",
        "mul": "*",
        "times": "*",
        "div": "/",
        "slash": "/",
        "forward_slash": "/",
        "lparen": "(",
        "rparen": ")",
    }
    return aliases.get(label, label if label else None)


def _to_grayscale(image_input):
    array = np.ascontiguousarray(np.asarray(image_input))
    if array.ndim == 2:
        return array.astype(np.uint8)
    if array.ndim == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {array.shape}")


def _binarize_character(gray_image: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(gray_image.astype(np.uint8))
    if image.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    try:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
    except cv2.error:
        blurred = image.copy()

    try:
        _, otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    except cv2.error:
        otsu = (blurred > 0).astype(np.uint8) * 255

    if np.count_nonzero(otsu) == 0:
        try:
            _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            otsu = (image > 0).astype(np.uint8) * 255

    if np.count_nonzero(otsu) > (otsu.size // 2):
        otsu = cv2.bitwise_not(otsu)

    return otsu


def _center_on_canvas(binary_image: np.ndarray, canvas_size: int = 28) -> np.ndarray:
    ys, xs = np.where(binary_image > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = binary_image[y0:y1, x0:x1]

    h, w = crop.shape
    scale = min(20.0 / max(h, 1), 20.0 / max(w, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def preprocess_character(image_input) -> np.ndarray:
    gray = _to_grayscale(image_input)
    binary = _binarize_character(gray)
    return _center_on_canvas(binary)


def _augment_character(image: np.ndarray, label_name: str, rng: np.random.Generator) -> np.ndarray:
    canvas = image.astype(np.uint8)
    center = (14.0, 14.0)

    if label_name == "*":
        angle = rng.uniform(-20.0, 20.0)
        scale = rng.uniform(0.82, 1.18)
        shear = rng.uniform(-0.18, 0.18)
        noise_std = rng.uniform(4.0, 10.0)
    elif label_name == "/":
        angle = rng.uniform(-12.0, 12.0)
        scale = rng.uniform(0.88, 1.14)
        shear = rng.uniform(-0.10, 0.10)
        noise_std = rng.uniform(3.0, 8.0)
    else:
        angle = rng.uniform(-10.0, 10.0)
        scale = rng.uniform(0.90, 1.10)
        shear = rng.uniform(-0.08, 0.08)
        noise_std = rng.uniform(2.0, 6.0)

    tx = float(rng.integers(-3, 4))
    ty = float(rng.integers(-3, 4))

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[:, 2] += [tx, ty]
    matrix[0, 1] += shear
    transformed = cv2.warpAffine(
        canvas,
        matrix,
        (28, 28),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    if rng.random() < 0.35:
        kernel = np.ones((2, 2), dtype=np.uint8)
        if rng.random() < 0.5:
            transformed = cv2.dilate(transformed, kernel, iterations=1)
        else:
            transformed = cv2.erode(transformed, kernel, iterations=1)

    if rng.random() < 0.18:
        transformed = cv2.GaussianBlur(transformed, (3, 3), sigmaX=0.5)

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, transformed.shape)
        transformed = np.clip(transformed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return preprocess_character(transformed)


def _load_feedback_samples(class_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    corrections_dir = FEEDBACK_DIR / "corrections"
    if not corrections_dir.exists():
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    images: list[np.ndarray] = []
    labels: list[int] = []

    for label_dir in sorted(corrections_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        normalized_label = _normalize_feedback_label(label_dir.name)
        if normalized_label not in class_to_index:
            continue

        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            images.append(preprocess_character(image))
            labels.append(class_to_index[normalized_label])

    if not images:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    return np.stack(images).astype(np.uint8), np.asarray(labels, dtype=np.int64)


def _build_target_augmentations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_names: list[str],
    rng: np.random.Generator,
    duplication: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    aug_images: list[np.ndarray] = []
    aug_labels: list[int] = []

    for label_name, repeats in duplication.items():
        label_idx = class_names.index(label_name)
        samples = x_train[y_train == label_idx]
        for image in samples:
            for _ in range(repeats):
                aug_images.append(_augment_character(image, label_name, rng))
                aug_labels.append(label_idx)

    if not aug_images:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    return np.stack(aug_images).astype(np.uint8), np.asarray(aug_labels, dtype=np.int64)


def _build_feedback_augmentations(
    feedback_images: np.ndarray,
    feedback_labels: np.ndarray,
    class_names: list[str],
    rng: np.random.Generator,
    feedback_aug_repeats: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    if len(feedback_images) == 0:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    aug_images: list[np.ndarray] = []
    aug_labels: list[int] = []

    for image, label_idx in zip(feedback_images, feedback_labels):
        label_name = class_names[int(label_idx)]
        repeats = feedback_aug_repeats.get(label_name, 0)
        for _ in range(repeats):
            aug_images.append(_augment_character(image, label_name, rng))
            aug_labels.append(int(label_idx))

    if not aug_images:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    return np.stack(aug_images).astype(np.uint8), np.asarray(aug_labels, dtype=np.int64)


def _compute_group_macro_metric(
    per_class: dict[str, dict[str, float | int]],
    label_names: tuple[str, ...],
    key: str,
) -> float:
    values = [float(per_class[name][key]) for name in label_names if name in per_class]
    return float(np.mean(values)) if values else 0.0


def _score_hard_example(
    true_name: str,
    pred_name: str,
    confidence: float,
    margin: float,
    confidence_threshold: float,
    margin_threshold: float,
) -> float:
    score = max(0.0, confidence_threshold - confidence) * 8.0
    score += max(0.0, margin_threshold - margin) * 10.0

    if pred_name != true_name:
        score += 1.0
        if pred_name in CONFUSION_PARTNERS.get(true_name, set()):
            score += 0.75

    return score


def _mine_hard_examples(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_names: list[str],
    batch_size: int,
    confidence_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    focus_limits = {
        class_names.index(name): limit
        for name, limit in DEFAULT_HARD_EXAMPLE_LIMITS.items()
        if name in class_names and limit > 0
    }
    if not focus_limits:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    probabilities = model.predict(
        x_train.astype(np.float32) / 255.0,
        batch_size=batch_size,
        verbose=0,
    )
    y_pred = probabilities.argmax(axis=1)
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    top_confidence = top_two.max(axis=1)
    margins = top_confidence - top_two.min(axis=1)

    hard_indices: list[int] = []

    for label_idx, limit in focus_limits.items():
        label_mask = y_train == label_idx
        sample_indices = np.flatnonzero(label_mask)
        if len(sample_indices) == 0:
            continue

        label_name = class_names[label_idx]
        scored: list[tuple[float, int]] = []

        for sample_idx in sample_indices:
            pred_idx = int(y_pred[sample_idx])
            pred_name = class_names[pred_idx]
            confidence = float(top_confidence[sample_idx])
            margin = float(margins[sample_idx])
            misclassified = pred_idx != label_idx
            confusion_hit = pred_name in CONFUSION_PARTNERS.get(label_name, set())
            low_confidence = confidence < confidence_threshold
            narrow_margin = margin < margin_threshold

            if not (misclassified or confusion_hit or low_confidence or narrow_margin):
                continue

            score = _score_hard_example(
                label_name,
                pred_name,
                confidence,
                margin,
                confidence_threshold,
                margin_threshold,
            )
            if score <= 0.0:
                continue
            scored.append((score, int(sample_idx)))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [idx for _, idx in scored[:limit]]

        fallback_limit = DEFAULT_HARD_FALLBACK_LIMITS.get(label_name, 0)
        if len(selected) < fallback_limit:
            remaining = [
                (
                    float(top_confidence[sample_idx]),
                    float(margins[sample_idx]),
                    int(sample_idx),
                )
                for sample_idx in sample_indices
                if int(sample_idx) not in selected
            ]
            remaining.sort(key=lambda item: (item[0], item[1], item[2]))
            selected.extend(
                sample_idx
                for _, _, sample_idx in remaining[:fallback_limit - len(selected)]
            )

        hard_indices.extend(selected)

    if not hard_indices:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    unique_indices = np.unique(np.asarray(hard_indices, dtype=np.int64))
    return x_train[unique_indices].copy(), y_train[unique_indices].copy()


def _build_hard_example_augmentations(
    hard_images: np.ndarray,
    hard_labels: np.ndarray,
    class_names: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(hard_images) == 0:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    aug_images: list[np.ndarray] = []
    aug_labels: list[int] = []

    for image, label_idx in zip(hard_images, hard_labels):
        label_name = class_names[int(label_idx)]
        repeats = DEFAULT_HARD_AUG_REPEATS.get(label_name, 0)
        for _ in range(repeats):
            aug_images.append(_augment_character(image, label_name, rng))
            aug_labels.append(int(label_idx))

    if not aug_images:
        return np.empty((0, 28, 28), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    return np.stack(aug_images).astype(np.uint8), np.asarray(aug_labels, dtype=np.int64)


def _build_sample_weights(
    labels: np.ndarray,
    class_names: list[str],
    base_weight_map: dict[str, float],
    boost_feedback: bool = False,
) -> np.ndarray:
    weights = np.ones(len(labels), dtype=np.float32)
    class_weight_map = {
        class_names.index(name): value
        for name, value in base_weight_map.items()
        if name in class_names
    }
    for label_idx, value in class_weight_map.items():
        weights[labels == label_idx] = value

    if boost_feedback and len(weights) > 0:
        for idx, label_idx in enumerate(labels):
            label_name = class_names[int(label_idx)]
            weights[idx] = TARGET_FEEDBACK_WEIGHT if label_name in TARGET_LABEL_NAMES else NON_TARGET_FEEDBACK_WEIGHT

    return weights


def _build_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE

    if sample_weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels, sample_weights))

    if training:
        dataset = dataset.shuffle(min(len(images), 50000), seed=seed, reshuffle_each_iteration=True)

    def _normalize(image, label, weight=None):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=-1)
        label = tf.cast(label, tf.int32)
        if weight is None:
            return image, label
        return image, label, tf.cast(weight, tf.float32)

    if sample_weights is None:
        dataset = dataset.map(_normalize, num_parallel_calls=autotune)
    else:
        dataset = dataset.map(_normalize, num_parallel_calls=autotune)

    return dataset.batch(batch_size).prefetch(autotune)


def evaluate_model(
    model: keras.Model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_names: list[str],
    batch_size: int,
) -> EvalMetrics:
    probabilities = model.predict(x_val.astype(np.float32) / 255.0, batch_size=batch_size, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    accuracy = float(accuracy_score(y_val, y_pred))
    _, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_val,
        y_pred,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0,
    )

    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_val,
        y_pred,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "support": int(support[index]),
        }

    target_macro_f1 = _compute_group_macro_metric(per_class, TARGET_LABEL_NAMES, "f1")
    target_macro_recall = _compute_group_macro_metric(per_class, TARGET_LABEL_NAMES, "recall")
    boundary_macro_f1 = _compute_group_macro_metric(per_class, BOUNDARY_LABEL_NAMES, "f1")
    boundary_macro_recall = _compute_group_macro_metric(per_class, BOUNDARY_LABEL_NAMES, "recall")
    priority_score = float(
        0.40 * target_macro_f1 +
        0.15 * target_macro_recall +
        0.20 * boundary_macro_f1 +
        0.10 * boundary_macro_recall +
        0.15 * accuracy
    )

    return EvalMetrics(
        accuracy=accuracy,
        macro_f1=float(macro_f1),
        macro_recall=float(macro_recall),
        target_macro_f1=target_macro_f1,
        target_macro_recall=target_macro_recall,
        boundary_macro_f1=boundary_macro_f1,
        boundary_macro_recall=boundary_macro_recall,
        priority_score=priority_score,
        per_class=per_class,
    )


class PriorityCheckpoint(keras.callbacks.Callback):
    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        class_names: list[str],
        batch_size: int,
        baseline: EvalMetrics,
        candidate_path: Path,
        csv_path: Path,
    ) -> None:
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.class_names = class_names
        self.batch_size = batch_size
        self.candidate_path = candidate_path
        self.csv_path = csv_path
        self.best_metrics = baseline
        self.improved = False
        self._initialized_csv = False

    def _write_row(self, row: dict) -> None:
        fieldnames = list(row.keys())
        write_header = not self.csv_path.exists() or not self._initialized_csv
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._initialized_csv = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = evaluate_model(
            self.model,
            self.x_val,
            self.y_val,
            self.class_names,
            self.batch_size,
        )

        row = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "macro_f1": metrics.macro_f1,
            "macro_recall": metrics.macro_recall,
            "target_macro_f1": metrics.target_macro_f1,
            "target_macro_recall": metrics.target_macro_recall,
            "boundary_macro_f1": metrics.boundary_macro_f1,
            "boundary_macro_recall": metrics.boundary_macro_recall,
            "priority_score": metrics.priority_score,
            "slash_f1": metrics.per_class["/"]["f1"],
            "slash_recall": metrics.per_class["/"]["recall"],
            "star_f1": metrics.per_class["*"]["f1"],
            "star_recall": metrics.per_class["*"]["recall"],
        }
        self._write_row(row)

        improved = metrics.priority_score > (self.best_metrics.priority_score + 1e-6)
        if improved:
            self.best_metrics = metrics
            self.improved = True
            self.model.save(self.candidate_path)

        print(
            f"[EPOCH {epoch + 1}] val_acc={metrics.accuracy:.4f} "
            f"target_f1={metrics.target_macro_f1:.4f} "
            f"target_recall={metrics.target_macro_recall:.4f} "
            f"priority={metrics.priority_score:.4f}"
        )


def _configure_trainable_layers(model: keras.Model, head_only: bool) -> None:
    if not head_only:
        for layer in model.layers:
            layer.trainable = True
        return

    trainable_names = {"flatten", "dense", "dropout_3", "dense_1"}
    for layer in model.layers:
        layer.trainable = layer.name in trainable_names


def _promote_candidate(candidate_path: Path, baseline_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = MODELS_DIR / f"best_expression_model.before_retrain_{timestamp}.keras"
    shutil.copy2(baseline_path, backup_path)
    shutil.copy2(candidate_path, baseline_path)
    print(f"[SAVE] Backed up baseline model to {backup_path}")
    print(f"[SAVE] Promoted retrained model to {baseline_path}")


def _print_metric_summary(label: str, metrics: EvalMetrics) -> None:
    print(
        f"{label}: accuracy={metrics.accuracy:.4f}, "
        f"macro_f1={metrics.macro_f1:.4f}, "
        f"target_f1={metrics.target_macro_f1:.4f}, "
        f"boundary_f1={metrics.boundary_macro_f1:.4f}, "
        f"target_recall={metrics.target_macro_recall:.4f}, "
        f"priority={metrics.priority_score:.4f}"
    )
    for name in FOCUS_LABEL_NAMES:
        item = metrics.per_class[name]
        print(
            f"  {name}: precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, "
            f"f1={item['f1']:.4f}, support={item['support']}"
        )


def _write_summary(
    class_names: list[str],
    train_sample_count: int,
    val_sample_count: int,
    epochs_requested: int,
    epochs_ran: int,
    learning_rate: float,
    baseline: EvalMetrics,
    final_metrics: EvalMetrics,
    promoted: bool,
    candidate_path: Path,
) -> None:
    payload = {
        "class_names": class_names,
        "dataset_dir": "model\\artifacts\\dataset",
        "model_path": "model\\artifacts\\models\\best_expression_model.keras",
        "candidate_model_path": str(candidate_path.relative_to(CURRENT_DIR)).replace("/", "\\"),
        "epochs_requested": epochs_requested,
        "epochs_ran": epochs_ran,
        "learning_rate": learning_rate,
        "train_samples": train_sample_count,
        "val_samples": val_sample_count,
        "promoted_candidate": promoted,
        "baseline_metrics": asdict(baseline),
        "final_metrics": asdict(final_metrics),
        "best_val_accuracy": final_metrics.accuracy,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the runtime Keras expression model.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--head-only", dest="head_only", action="store_true")
    parser.add_argument("--full-finetune", dest="head_only", action="store_false")
    parser.set_defaults(head_only=True)
    parser.add_argument("--star-dup", type=int, default=DEFAULT_TARGET_DUPLICATION["*"])
    parser.add_argument("--slash-dup", type=int, default=DEFAULT_TARGET_DUPLICATION["/"])
    parser.add_argument("--star-weight", type=float, default=DEFAULT_TARGET_BASE_WEIGHTS["*"])
    parser.add_argument("--slash-weight", type=float, default=DEFAULT_TARGET_BASE_WEIGHTS["/"])
    parser.add_argument("--one-weight", type=float, default=DEFAULT_TARGET_BASE_WEIGHTS["1"])
    parser.add_argument("--lparen-weight", type=float, default=DEFAULT_TARGET_BASE_WEIGHTS["("])
    parser.add_argument("--rparen-weight", type=float, default=DEFAULT_TARGET_BASE_WEIGHTS[")"])
    parser.add_argument("--feedback-aug-star", type=int, default=DEFAULT_FEEDBACK_AUG_REPEATS["*"])
    parser.add_argument("--feedback-aug-slash", type=int, default=DEFAULT_FEEDBACK_AUG_REPEATS["/"])
    parser.add_argument("--hard-confidence-threshold", type=float, default=DEFAULT_HARD_CONFIDENCE_THRESHOLD)
    parser.add_argument("--hard-margin-threshold", type=float, default=DEFAULT_HARD_MARGIN_THRESHOLD)
    parser.add_argument("--hard-weight-multiplier", type=float, default=DEFAULT_HARD_WEIGHT_MULTIPLIER)
    parser.add_argument("--disable-hard-mining", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names()
    class_to_index = {name: index for index, name in enumerate(class_names)}
    rng = np.random.default_rng(args.seed)
    duplication = {
        "*": max(0, args.star_dup),
        "/": max(0, args.slash_dup),
    }
    feedback_aug_repeats = {
        "*": max(0, args.feedback_aug_star),
        "/": max(0, args.feedback_aug_slash),
    }
    base_weight_map = {
        "*": float(args.star_weight),
        "/": float(args.slash_weight),
        "1": float(args.one_weight),
        "(": float(args.lparen_weight),
        ")": float(args.rparen_weight),
    }

    x_train = np.load(DATASET_DIR / "x_train.npy")
    y_train = np.load(DATASET_DIR / "y_train.npy")
    x_val = np.load(DATASET_DIR / "x_val.npy")
    y_val = np.load(DATASET_DIR / "y_val.npy")

    baseline_model = keras.models.load_model(MODEL_PATH, compile=False)
    baseline_metrics = evaluate_model(baseline_model, x_val, y_val, class_names, args.batch_size)
    _print_metric_summary("Baseline", baseline_metrics)

    feedback_images, feedback_labels = _load_feedback_samples(class_to_index)
    target_aug_images, target_aug_labels = _build_target_augmentations(
        x_train,
        y_train,
        class_names,
        rng,
        duplication,
    )
    feedback_aug_images, feedback_aug_labels = _build_feedback_augmentations(
        feedback_images,
        feedback_labels,
        class_names,
        rng,
        feedback_aug_repeats,
    )
    if args.disable_hard_mining:
        hard_images = np.empty((0, 28, 28), dtype=np.uint8)
        hard_labels = np.empty((0,), dtype=np.int64)
    else:
        hard_images, hard_labels = _mine_hard_examples(
            baseline_model,
            x_train,
            y_train,
            class_names,
            args.batch_size,
            confidence_threshold=float(args.hard_confidence_threshold),
            margin_threshold=float(args.hard_margin_threshold),
        )
    hard_aug_images, hard_aug_labels = _build_hard_example_augmentations(
        hard_images,
        hard_labels,
        class_names,
        rng,
    )

    base_weights = _build_sample_weights(y_train, class_names, base_weight_map)
    aug_weights = _build_sample_weights(target_aug_labels, class_names, base_weight_map)
    feedback_weights = _build_sample_weights(feedback_labels, class_names, base_weight_map, boost_feedback=True)
    feedback_aug_weights = _build_sample_weights(
        feedback_aug_labels,
        class_names,
        base_weight_map,
        boost_feedback=True,
    )
    hard_weights = _build_sample_weights(hard_labels, class_names, base_weight_map) * float(args.hard_weight_multiplier)
    hard_aug_weights = _build_sample_weights(hard_aug_labels, class_names, base_weight_map) * float(args.hard_weight_multiplier)

    image_parts = [x_train]
    label_parts = [y_train]
    weight_parts = [base_weights]

    if len(target_aug_images):
        image_parts.append(target_aug_images)
        label_parts.append(target_aug_labels)
        weight_parts.append(aug_weights)

    if len(feedback_images):
        image_parts.append(feedback_images)
        label_parts.append(feedback_labels)
        weight_parts.append(feedback_weights)

    if len(feedback_aug_images):
        image_parts.append(feedback_aug_images)
        label_parts.append(feedback_aug_labels)
        weight_parts.append(feedback_aug_weights)

    if len(hard_images):
        image_parts.append(hard_images)
        label_parts.append(hard_labels)
        weight_parts.append(hard_weights)

    if len(hard_aug_images):
        image_parts.append(hard_aug_images)
        label_parts.append(hard_aug_labels)
        weight_parts.append(hard_aug_weights)

    full_x_train = np.concatenate(image_parts, axis=0)
    full_y_train = np.concatenate(label_parts, axis=0)
    full_sample_weights = np.concatenate(weight_parts, axis=0)

    permutation = rng.permutation(len(full_x_train))
    full_x_train = full_x_train[permutation]
    full_y_train = full_y_train[permutation]
    full_sample_weights = full_sample_weights[permutation]

    train_dataset = _build_dataset(
        full_x_train,
        full_y_train,
        full_sample_weights,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
    )
    val_dataset = _build_dataset(
        x_val,
        y_val,
        None,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
    )

    model = keras.models.load_model(MODEL_PATH, compile=False)
    _configure_trainable_layers(model, head_only=args.head_only)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    candidate_path = MODELS_DIR / "best_expression_model.retrain_candidate.keras"
    if candidate_path.exists():
        candidate_path.unlink()
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    priority_callback = PriorityCheckpoint(
        x_val=x_val,
        y_val=y_val,
        class_names=class_names,
        batch_size=args.batch_size,
        baseline=baseline_metrics,
        candidate_path=candidate_path,
        csv_path=LOG_PATH,
    )

    callbacks = [
        priority_callback,
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(
        f"[TRAIN] train_samples={len(full_x_train)} "
        f"(base={len(x_train)}, target_aug={len(target_aug_images)}, "
        f"feedback={len(feedback_images)}, feedback_aug={len(feedback_aug_images)}, "
        f"hard={len(hard_images)}, hard_aug={len(hard_aug_images)})"
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    promoted = False
    final_metrics = baseline_metrics

    if priority_callback.improved and candidate_path.exists():
        candidate_model = keras.models.load_model(candidate_path, compile=False)
        candidate_metrics = evaluate_model(candidate_model, x_val, y_val, class_names, args.batch_size)
        _print_metric_summary("Candidate", candidate_metrics)

        accuracy_floor = baseline_metrics.accuracy - 0.0025
        if (
            candidate_metrics.priority_score > baseline_metrics.priority_score + 1e-6
            and candidate_metrics.accuracy >= accuracy_floor
        ):
            _promote_candidate(candidate_path, MODEL_PATH)
            promoted = True
            final_metrics = candidate_metrics
        else:
            print("[SAVE] Candidate did not beat baseline strongly enough. Baseline model kept.")
            final_metrics = candidate_metrics
    else:
        print("[SAVE] No epoch beat the baseline priority score. Baseline model kept.")

    _write_summary(
        class_names=class_names,
        train_sample_count=len(full_x_train),
        val_sample_count=len(x_val),
        epochs_requested=args.epochs,
        epochs_ran=len(history.history.get("loss", [])),
        learning_rate=args.learning_rate,
        baseline=baseline_metrics,
        final_metrics=final_metrics,
        promoted=promoted,
        candidate_path=candidate_path,
    )


if __name__ == "__main__":
    main()
