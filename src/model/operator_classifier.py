import os
import sys
import random
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torchvision.datasets as datasets

CURRENT_DIR = Path(__file__).resolve().parent # src/model
MODEL_DIR = CURRENT_DIR # src/model
SRC_DIR = MODEL_DIR.parent # src
PROJECT_ROOT = SRC_DIR.parent # project root
if str(SRC_DIR) not in sys.path: # Ensure src is in path for absolute imports
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.preprocessing import preprocess, preprocess_for_keras # preprocessing is still in src/preprocessing


def _resolve_runtime_path(name):
    cwd_path = os.path.abspath(name)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(CURRENT_DIR, name)


def _project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


TORCH_MODEL_PATH = MODEL_DIR / "model_combined.pth"
KERAS_MODEL_PATH = MODEL_DIR / "artifacts" / "models" / "best_expression_model.keras"
TRAINING_SUMMARY_PATH = MODEL_DIR / "artifacts" / "models" / "training_summary.json"
SYNTH_DATA_DIR = MODEL_DIR / "synthetic_operators"
REAL_OPERATOR_DIR = MODEL_DIR / "extracted_images"
DATA_DIR = PROJECT_ROOT / "data"

SAMPLES_PER_CLASS = 5000
NUM_CLASSES = 16
EPOCHS = 6
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
SAMPLES_PER_EPOCH = 64000
SYNTHETIC_DATASET_VERSION = "v2"

INDEX_TO_CHAR = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "+", 11: "-", 12: "*", 13: "/", 14: "(", 15: ")",
}
CHAR_TO_INDEX = {value: key for key, value in INDEX_TO_CHAR.items()}


def _default_class_names():
    return [INDEX_TO_CHAR[index] for index in range(NUM_CLASSES)]


def _load_runtime_class_names():
    if not os.path.exists(TRAINING_SUMMARY_PATH):
        return _default_class_names()

    try:
        with open(TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return _default_class_names()

    class_names = summary.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != NUM_CLASSES:
        return _default_class_names()

    return [str(name) for name in class_names]

REAL_OPERATOR_FOLDERS = {
    "+": 10,
    "-": 11,
    "times": 12,
    "forward_slash": 13,
    "(": 14,
    ")": 15,
}

SYNTH_OPERATOR_FOLDERS = {
    "plus": 10,
    "minus": 11,
    "mul": 12,
    "div": 13,
    "lparen": 14,
    "rparen": 15,
}

AMBIGUOUS_LABELS = {1, 11, 13, 14, 15}


def _draw_plus(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 255, thickness)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), 255, thickness)


def _draw_minus(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 255, thickness)


def _draw_multiply(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy - arm), (cx + arm, cy + arm), 255, thickness)
    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), 255, thickness)


def _draw_divide(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), 255, thickness)


def _draw_left_paren(img, cx, cy, size, thickness):
    axes = (max(2, size // 6), max(7, size // 2))
    cv2.ellipse(img, (cx + 1, cy), axes, 0, 78, 282, 255, thickness)


def _draw_right_paren(img, cx, cy, size, thickness):
    axes = (max(2, size // 6), max(7, size // 2))
    cv2.ellipse(img, (cx - 1, cy), axes, 0, -102, 102, 255, thickness)


DRAW_FN = {
    "+": _draw_plus,
    "-": _draw_minus,
    "*": _draw_multiply,
    "/": _draw_divide,
    "(": _draw_left_paren,
    ")": _draw_right_paren,
}


def _synthetic_version_path():
    return os.path.join(SYNTH_DATA_DIR, ".dataset_version")


def _synthetic_dataset_is_current():
    version_path = _synthetic_version_path()
    if not os.path.isfile(version_path):
        return False

    try:
        with open(version_path, "r", encoding="utf-8") as handle:
            return handle.read().strip() == SYNTHETIC_DATASET_VERSION
    except OSError:
        return False


def _clear_operator_folder(folder_path):
    if not os.path.isdir(folder_path):
        return

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path):
            os.remove(path)


def _sample_operator_params(symbol):
    if symbol == "+":
        return {
            "cx": 14 + np.random.randint(-2, 3),
            "cy": 14 + np.random.randint(-2, 3),
            "size": np.random.randint(10, 18),
            "thickness": np.random.randint(1, 4),
            "angle": np.random.uniform(-10, 10),
            "noise_std": 7.0,
        }
    if symbol == "-":
        return {
            "cx": 14 + np.random.randint(-2, 3),
            "cy": 14 + np.random.randint(-1, 2),
            "size": np.random.randint(10, 18),
            "thickness": np.random.randint(1, 3),
            "angle": np.random.uniform(-4, 4),
            "noise_std": 5.0,
        }
    if symbol == "*":
        return {
            "cx": 14 + np.random.randint(-2, 3),
            "cy": 14 + np.random.randint(-2, 3),
            "size": np.random.randint(10, 17),
            "thickness": np.random.randint(1, 3),
            "angle": np.random.uniform(-8, 8),
            "noise_std": 6.0,
        }
    if symbol == "/":
        return {
            "cx": 14 + np.random.randint(-1, 2),
            "cy": 14 + np.random.randint(-1, 2),
            "size": np.random.randint(13, 19),
            "thickness": np.random.randint(1, 3),
            "angle": np.random.uniform(-5, 5),
            "noise_std": 4.0,
        }
    if symbol in {"(", ")"}:
        return {
            "cx": 14 + np.random.randint(-1, 2),
            "cy": 14 + np.random.randint(-2, 3),
            "size": np.random.randint(14, 19),
            "thickness": np.random.randint(1, 3),
            "angle": np.random.uniform(-6, 6),
            "noise_std": 4.0,
        }

    return {
        "cx": 14,
        "cy": 14,
        "size": 14,
        "thickness": 2,
        "angle": 0.0,
        "noise_std": 5.0,
    }


def generate_operator_dataset():
    os.makedirs(SYNTH_DATA_DIR, exist_ok=True)

    for symbol, draw_fn in DRAW_FN.items():
        safe_name = {
            "+": "plus",
            "-": "minus",
            "*": "mul",
            "/": "div",
            "(": "lparen",
            ")": "rparen",
        }[symbol]
        folder = os.path.join(SYNTH_DATA_DIR, safe_name)
        os.makedirs(folder, exist_ok=True)
        _clear_operator_folder(folder)

        print(f"[GEN] Generating {SAMPLES_PER_CLASS} samples for '{symbol}'")
        for i in range(SAMPLES_PER_CLASS):
            img = np.zeros((28, 28), dtype=np.uint8)
            params = _sample_operator_params(symbol)
            cx = params["cx"]
            cy = params["cy"]
            size = params["size"]
            thickness = params["thickness"]

            draw_fn(img, cx, cy, size, thickness)

            angle = params["angle"]
            matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, matrix, (28, 28))

            noise = np.random.normal(0, params["noise_std"], img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            shift = np.random.randint(-10, 11)
            img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(folder, f"{i:05d}.png"), img)

    with open(_synthetic_version_path(), "w", encoding="utf-8") as handle:
        handle.write(SYNTHETIC_DATASET_VERSION)

    print("[GEN] Synthetic operator dataset generation complete.")


_cached_runtime_model = None
_runtime_class_names = None


def _load_runtime_model():
    global _cached_runtime_model
    if _cached_runtime_model is None:
        if not os.path.exists(KERAS_MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{KERAS_MODEL_PATH}' not found." # KERAS_MODEL_PATH is updated
            )

        import keras

        model = keras.models.load_model(KERAS_MODEL_PATH)
        output_shape = getattr(model, "output_shape", None)
        if output_shape is None or output_shape[-1] != NUM_CLASSES:
            raise ValueError(
                "The runtime Keras model output does not match the expected "
                f"{NUM_CLASSES} classes: {output_shape}"
            )
        _cached_runtime_model = model

    return _cached_runtime_model


def _get_runtime_class_names():
    global _runtime_class_names
    if _runtime_class_names is None:
        _runtime_class_names = _load_runtime_class_names()
    return _runtime_class_names


def _looks_like_normalized_roi(image_input):
    if image_input is None:
        return False

    if len(image_input.shape) == 3 and image_input.shape[2] == 3:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image_input.copy()

    if gray.shape != (28, 28):
        return False

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    edges = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    edge_mean = float(edges.mean())
    bright_ratio = np.count_nonzero(gray >= 180) / float(gray.size)
    dark_ratio = np.count_nonzero(gray <= 40) / float(gray.size)

    return edge_mean <= 45.0 and bright_ratio >= 0.01 and dark_ratio >= 0.35


def _prepare_keras_batch(images_28x28, normalized=None):
    if not images_28x28:
        return np.empty((0, 28, 28, 1), dtype=np.float32)

    processed = []
    for image_28x28 in images_28x28:
        assume_normalized = normalized
        if assume_normalized is None:
            assume_normalized = _looks_like_normalized_roi(image_28x28)
        processed.append(preprocess_for_keras(image_28x28, assume_normalized=assume_normalized)[0])

    return np.stack(processed, axis=0)


def predict_characters_top_k(images_28x28, normalized=None, top_k=5):
    model = _load_runtime_model()
    class_names = _get_runtime_class_names()
    if not images_28x28:
        return []

    batch = _prepare_keras_batch(images_28x28, normalized=normalized)
    probabilities = model.predict(batch, verbose=0)
    top_k = max(1, min(int(top_k), len(class_names)))

    results = []
    for row in probabilities:
        ranked_indices = np.argsort(row)[::-1][:top_k]
        results.append([
            {
                "char": class_names[int(index)],
                "conf": float(row[int(index)]),
            }
            for index in ranked_indices
        ])

    return results


def predict_character_top_k(image_28x28, normalized=None, top_k=5):
    results = predict_characters_top_k([image_28x28], normalized=normalized, top_k=top_k)
    return results[0] if results else []


def predict_character(image_28x28, normalized=None):
    model = _load_runtime_model()
    batch = _prepare_keras_batch([image_28x28], normalized=normalized)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    class_names = _get_runtime_class_names()
    return class_names[predicted_index], confidence


def export_datasets_to_numpy(val_ratio=0.1, seed=42):
    """
    Exports the raw datasets (MNIST, EMNIST, and Operators) into combined,
    shuffled NumPy arrays for use by the Keras pipeline.
    """
    output_dir = MODEL_DIR / "artifacts" / "dataset" # output_dir is updated
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isdir(SYNTH_DATA_DIR) or not _synthetic_dataset_is_current():
        generate_operator_dataset()

    print("[EXPORT] Building datasets for NumPy export (No Augmentation)...")
    np.random.seed(seed)
    
    images_list = []
    labels_list = []
    
    print("[EXPORT] Loading MNIST...")
    mnist = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    images_list.append(mnist.data.numpy())
    labels_list.append(mnist.targets.numpy())
    
    print("[EXPORT] Loading EMNIST...")
    emnist = datasets.EMNIST(root=DATA_DIR, split="digits", train=True, download=True)
    emnist_imgs = emnist.data.numpy()
    emnist_imgs = np.transpose(emnist_imgs, (0, 2, 1)) # Fix orientation
    images_list.append(emnist_imgs)
    labels_list.append(emnist.targets.numpy())
    
    print("[EXPORT] Loading Custom Operators...")
    for folder_to_class, root_dir in [(REAL_OPERATOR_FOLDERS, REAL_OPERATOR_DIR), (SYNTH_OPERATOR_FOLDERS, SYNTH_DATA_DIR)]:
        for folder_name, class_idx in folder_to_class.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path): continue
            
            for name in os.listdir(folder_path):
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    image = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        normalized = preprocess(image)
                        images_list.append(np.expand_dims(normalized, axis=0))
                        labels_list.append(np.array([class_idx]))

    print("[EXPORT] Concatenating and shuffling arrays...")
    x_all = np.concatenate(images_list, axis=0).astype(np.uint8)
    y_all = np.concatenate(labels_list, axis=0).astype(np.int64)
    
    indices = np.random.permutation(len(x_all))
    x_all = x_all[indices]
    y_all = y_all[indices]
    
    val_size = int(len(x_all) * val_ratio)
    x_val = x_all[:val_size]
    y_val = y_all[:val_size]
    x_train = x_all[val_size:]
    y_train = y_all[val_size:]
    
    print(f"[EXPORT] Saving to '{output_dir}' ...")
    np.save(os.path.join(output_dir, "x_train.npy"), x_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "x_val.npy"), x_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    
    print(f"[EXPORT] Done! Train size: {len(x_train)}, Val size: {len(x_val)}")

if __name__ == '__main__':
    # train_combined_model()
    export_datasets_to_numpy()
