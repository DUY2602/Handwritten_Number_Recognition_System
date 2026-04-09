import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# VGG-style 8-layer CNN architecture used to train

def create_model(input_shape=(28, 28, 1), num_classes=16):

    model = keras.Sequential([
        keras.Input(shape=input_shape, name="input_layer"),
        
        # -- Built-in Data Augmentation Layer --
        # Dynamically applies random transformations during training to artificially expand 
        # the dataset and prevent the model from memorizing exact image orientations.
        layers.RandomRotation(0.1, name="random_rotation"),
        layers.RandomTranslation(0.1, 0.1, name="random_translation"),

        # -- Block 1: Low-Level Feature Extraction --
        # Extracts simple, fundamental features like edges and basic straight lines.
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv2d"),
        layers.BatchNormalization(name="batch_normalization"),
        layers.Activation("relu", name="activation"),
        
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv2d_1"),
        layers.BatchNormalization(name="batch_normalization_1"),
        layers.Activation("relu", name="activation_1"),
        
        layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"),
        layers.Dropout(0.25, name="dropout"),

        # -- Block 2: Mid-Level Feature Extraction --
        # Combines low-level features to understand shapes like curves and corners.
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2d_2"),
        layers.BatchNormalization(name="batch_normalization_2"),
        layers.Activation("relu", name="activation_2"),
        
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2d_3"),
        layers.BatchNormalization(name="batch_normalization_3"),
        layers.Activation("relu", name="activation_3"),
        
        layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
        layers.Dropout(0.25, name="dropout_1"),

        # -- Block 3: High-Level Feature Extraction --
        # Detects highly specific semantic parts of mathematical characters and operators.
        layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv2d_4"),
        layers.BatchNormalization(name="batch_normalization_4"),
        layers.Activation("relu", name="activation_4"),
        
        layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv2d_5"),
        layers.BatchNormalization(name="batch_normalization_5"),
        layers.Activation("relu", name="activation_5"),
        
        layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
        layers.Dropout(0.25, name="dropout_2"),

        # -- Classifier (Fully-Connected Network) --
        # Flattens the 2D feature maps into a 1D vector and uses dense layers to 
        # map these extracted features to the final 16 output classes.
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense"),
        layers.Dropout(0.4, name="dropout_3"),
        layers.Dense(num_classes, activation="softmax", name="dense_1"),
    ])
    
    return model

if __name__ == "__main__":
    print("[INFO] Building VGG-style 8-Layer CNN Baseline Model...")
    model = create_model()
    model.summary()

import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "artifacts" / "dataset"

print("[INFO] Loading dataset from .npy files in 'artifacts/dataset'...")
x_train = np.load(DATASET_DIR / "x_train.npy")
y_train = np.load(DATASET_DIR / "y_train.npy")
x_val = np.load(DATASET_DIR / "x_val.npy")
y_val = np.load(DATASET_DIR / "y_val.npy")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"[INFO] Training model on {len(x_train)} samples...")
model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_val, y_val))

model.save(CURRENT_DIR / "artifacts" / "models" / "best_expression_model.keras")
print("[INFO] Baseline model saved successfully!")
