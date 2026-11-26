#!/usr/bin/env python3
"""Train MobileNetV2 DR classifier and export models/*.h5."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, FULL_DIR, MODEL_PATH, SEED, TRAIN_DIR
from src.insights import compute_insights
from src.model import build_model, compile_model, save_model, unfreeze_top
from src.preprocessing import preprocess_image


def load_dir(root: Path):
    xs, ys = [], []
    for class_id, name in CLASS_NAMES.items():
        folder = root / name
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            xs.append(preprocess_image(p))
            ys.append(class_id)
    if not xs:
        raise FileNotFoundError(f"No images under {root}")
    return np.stack(xs), np.array(ys, dtype=np.int32)


def augmenter(seed: int = SEED) -> keras.Sequential:
    """Train-time only. Fundus photos have no canonical orientation, so flips and
    full rotations are label-preserving; zoom/brightness/contrast mimic camera variation."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomZoom((-0.1, 0.1), fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomBrightness(0.1, value_range=(0.0, 1.0), seed=seed),
            layers.RandomContrast(0.1, seed=seed),
        ],
        name="augment",
    )


class AugmentedBatches(keras.utils.PyDataset):
    """Shuffled, augmented batches from in-memory arrays (avoids tf.data's 2 GB constant limit).

    Class weights are passed as per-sample weights so imbalance handling matches the
    non-augmented path.
    """

    def __init__(self, x, y, batch_size, weights, seed=SEED, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y, self.batch_size = x, y, batch_size
        self.sample_w = np.array([weights[int(c)] for c in y], dtype=np.float32)
        self.aug = augmenter(seed)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size : (i + 1) * self.batch_size]
        xb = np.clip(np.asarray(self.aug(self.x[idx], training=True)), 0.0, 1.0)
        return xb, self.y[idx], self.sample_w[idx]

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.x))


def class_weights(y: np.ndarray) -> dict:
    counts = Counter(y.tolist())
    total = len(y)
    n_classes = len(counts)
    return {c: total / (n_classes * n) for c, n in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=FULL_DIR if (FULL_DIR / "No_DR").exists() else TRAIN_DIR,
    )
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-ft", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-per-class", type=int, default=0, help="0 = all")
    parser.add_argument("--patience", type=int, default=3, help="early-stopping patience in epochs (LR halves after patience//2)")
    parser.add_argument("--augment", action="store_true", help="random flips/rotation/zoom/brightness/contrast on training batches")
    parser.add_argument("--out", type=Path, default=MODEL_PATH, help="where to save the .h5")
    parser.add_argument(
        "--metrics-out", type=Path, default=None,
        help="metrics JSON path (default: models/metrics.json, or <out>.metrics.json when --out is set)",
    )
    args = parser.parse_args()
    # Experiments (custom --out) must not touch production metrics or the insights cache
    is_production = args.out.resolve() == MODEL_PATH.resolve()
    if args.metrics_out is None:
        args.metrics_out = ROOT / "models" / "metrics.json" if is_production else args.out.with_suffix(".metrics.json")

    x, y = load_dir(args.data)
    if args.max_per_class:
        keep = []
        counts = Counter()
        for i, label in enumerate(y):
            if counts[label] < args.max_per_class:
                keep.append(i)
                counts[label] += 1
        x, y = x[keep], y[keep]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    weights = class_weights(y_train)
    print("class_weights", weights)
    if args.augment:
        print("augmentation: on")
        train_data = dict(x=AugmentedBatches(x_train, y_train, args.batch_size, weights))
    else:
        train_data = dict(x=x_train, y=y_train, batch_size=args.batch_size, class_weight=weights)

    model = build_model(trainable_base=False)
    compile_model(model, learning_rate=1e-3)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2)),
    ]

    print("=== Phase 1: frozen backbone ===")
    model.fit(
        **train_data,
        validation_data=(x_val, y_val),
        epochs=args.epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    print("=== Phase 2: fine-tune top MobileNet layers ===")
    unfreeze_top(model, n_layers=40)
    compile_model(model, learning_rate=1e-5)
    history = model.fit(
        **train_data,
        validation_data=(x_val, y_val),
        epochs=args.epochs_ft,
        callbacks=callbacks,
        verbose=1,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)
    print(f"Saved {args.out}")

    probs = model.predict(x_val, verbose=0)
    y_pred = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision_macro": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "val_loss": float(history.history["val_loss"][-1]),
        "classification_report": classification_report(
            y_val,
            y_pred,
            target_names=[CLASS_NAMES[i] for i in range(5)],
            zero_division=0,
            output_dict=True,
        ),
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "val_loss")}, indent=2))

    if is_production:
        compute_insights(TRAIN_DIR if TRAIN_DIR.exists() else args.data)
        print("insights cached")


if __name__ == "__main__":
    main()
