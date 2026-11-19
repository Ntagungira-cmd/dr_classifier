"""MobileNetV2 transfer-learning model for 5-class DR grading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE, MODEL_PATH, NUM_CLASSES


def build_model(
    *,
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    trainable_base: bool = False,
    dropout: float = 0.3,
    l2: float = 1e-4,
) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    inputs = keras.Input(shape=(img_size, img_size, 3))
    # MobileNetV2 expects [-1, 1]
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2),
    )(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="dr_mobilenetv2")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseCategoricalAccuracy(name="sparse_acc"),
        ],
    )
    return model


def load_model(path: Optional[Path] = None) -> keras.Model:
    path = Path(path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train first.")
    return keras.models.load_model(path)


def save_model(model: keras.Model, path: Optional[Path] = None) -> Path:
    path = Path(path or MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def unfreeze_top(model: keras.Model, n_layers: int = 40) -> None:
    """Unfreeze last n layers of the MobileNetV2 backbone for fine-tuning."""
    base = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenet" in layer.name.lower():
            base = layer
            break
    if base is None:
        return
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
