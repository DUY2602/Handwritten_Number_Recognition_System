import os
import sys
import random
import json
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.preprocessing import preprocess, preprocess_for_keras


def _resolve_runtime_path(name):
    cwd_path = os.path.abspath(name)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(CURRENT_DIR, name)


def _project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


TORCH_MODEL_PATH = _resolve_runtime_path("model_combined.pth")
KERAS_MODEL_PATH = _project_path(
    "src", "model", "artifacts", "models", "best_expression_model.keras"
)
TRAINING_SUMMARY_PATH = _project_path(
    "src", "model", "artifacts", "models", "training_summary.json"
)
SYNTH_DATA_DIR = _resolve_runtime_path("synthetic_operators")
REAL_OPERATOR_DIR = _resolve_runtime_path(os.path.join("src", "model", "extracted_images"))
DATA_DIR = _resolve_runtime_path("data")

SAMPLES_PER_CLASS = 5000
NUM_CLASSES = 16
EPOCHS = 6
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
SAMPLES_PER_EPOCH = 64000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class LabelAwareTrainTransform:
    def __init__(self):
        self.default = transforms.Compose([
            transforms.RandomAffine(
                degrees=12,
                translate=(0.1, 0.1),
                scale=(0.93, 1.08),
                shear=5,
                fill=0,
            ),
            transforms.RandomPerspective(distortion_scale=0.16, p=0.18, fill=0),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))],
                p=0.18,
            ),
        ])
        self.ambiguous = transforms.Compose([
            transforms.RandomAffine(
                degrees=6,
                translate=(0.06, 0.08),
                scale=(0.97, 1.04),
                shear=1.5,
                fill=0,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.45))],
                p=0.08,
            ),
        ])
        self.minus = transforms.Compose([
            transforms.RandomAffine(
                degrees=3,
                translate=(0.08, 0.04),
                scale=(0.96, 1.04),
                shear=0,
                fill=0,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.35))],
                p=0.05,
            ),
        ])

    def __call__(self, tensor, label):
        label = int(label)
        if label == 11:
            return self.minus(tensor)
        if label in AMBIGUOUS_LABELS:
            return self.ambiguous(tensor)
        return self.default(tensor)


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


class TorchVisionDigitDataset(Dataset):
    def __init__(self, dataset, transform=None, fix_orientation=False):
        self.dataset = dataset
        self.transform = transform
        self.fix_orientation = fix_orientation

        raw_targets = dataset.targets
        if hasattr(raw_targets, "tolist"):
            raw_targets = raw_targets.tolist()
        self.labels = [int(label) for label in raw_targets]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        array = np.array(image, dtype=np.uint8)

        if self.fix_orientation:
            # EMNIST digits are transposed relative to MNIST.
            array = np.ascontiguousarray(array.T)

        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.transform is not None:
            try:
                tensor = self.transform(tensor, int(label))
            except TypeError:
                tensor = self.transform(tensor)

        return tensor, int(label)


class ImageFolderCharacterDataset(Dataset):
    def __init__(
        self,
        root_dir,
        folder_to_class,
        transform=None,
        split="train",
        train_ratio=0.9,
        seed=42,
    ):
        self.transform = transform
        self.samples = []
        self.labels = []

        splitter = random.Random(seed)
        for folder_name, class_idx in folder_to_class.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            files = [
                os.path.join(folder_path, name)
                for name in os.listdir(folder_path)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            files.sort()
            splitter.shuffle(files)

            if split in {"train", "val"}:
                cutoff = max(1, int(len(files) * train_ratio))
                selected = files[:cutoff] if split == "train" else files[cutoff:]
            else:
                selected = files

            for path in selected:
                self.samples.append((path, class_idx))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        normalized = preprocess(image)
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.transform is not None:
            try:
                tensor = self.transform(tensor, int(label))
            except TypeError:
                tensor = self.transform(tensor)

        return tensor, int(label)


class OperatorCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _build_train_transform():
    return LabelAwareTrainTransform()


def _build_training_datasets():
    transform = _build_train_transform()

    mnist_train = TorchVisionDigitDataset(
        datasets.MNIST(root=DATA_DIR, train=True, download=True),
        transform=transform,
        fix_orientation=False,
    )
    emnist_train = TorchVisionDigitDataset(
        datasets.EMNIST(root=DATA_DIR, split="digits", train=True, download=True),
        transform=transform,
        fix_orientation=True,
    )
    real_operator_train = ImageFolderCharacterDataset(
        REAL_OPERATOR_DIR,
        REAL_OPERATOR_FOLDERS,
        transform=transform,
        split="train",
    )
    synthetic_operator_train = ImageFolderCharacterDataset(
        SYNTH_DATA_DIR,
        SYNTH_OPERATOR_FOLDERS,
        transform=transform,
        split="train",
    )

    return [
        mnist_train,
        emnist_train,
        real_operator_train,
        synthetic_operator_train,
    ]


def _build_balanced_loader(datasets_list):
    labels = []
    for dataset in datasets_list:
        labels.extend(dataset.labels)

    label_counts = Counter(labels)
    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
    for label, count in label_counts.items():
        class_weights[label] = len(labels) / float(NUM_CLASSES * count)

    sample_weights = [float(class_weights[label]) for label in labels]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=SAMPLES_PER_EPOCH,
        replacement=True,
    )

    loader = DataLoader(
        ConcatDataset(datasets_list),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )
    return loader, class_weights


def train_combined_model():
    if not os.path.isdir(SYNTH_DATA_DIR) or not _synthetic_dataset_is_current():
        generate_operator_dataset()
    else:
        print("[INFO] Synthetic operator dataset already exists.")

    train_datasets = _build_training_datasets()
    train_loader, class_weights = _build_balanced_loader(train_datasets)

    model = OperatorCNN(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(TORCH_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(TORCH_MODEL_PATH, map_location=DEVICE))
            print("[INFO] Loaded existing weights for fine-tuning.")
        except RuntimeError:
            print("[WARN] Existing weights do not match current architecture. Training from scratch.")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"[TRAIN] Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(1)
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()

        accuracy = 100.0 * correct / max(1, total)
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"  Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%")

    torch.save(model.state_dict(), TORCH_MODEL_PATH)
    print(f"[TRAIN] Model saved to '{TORCH_MODEL_PATH}'")

    global _cached_torch_model
    _cached_torch_model = model.eval()


_cached_torch_model = None
_cached_runtime_model = None
_runtime_class_names = None


def _load_runtime_model():
    global _cached_runtime_model
    if _cached_runtime_model is None:
        if not os.path.exists(KERAS_MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{KERAS_MODEL_PATH}' not found."
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


if __name__ == "__main__":
    train_combined_model()
