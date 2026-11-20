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
    args = parser.parse_args()

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

    model = build_model(trainable_base=False)
    compile_model(model, learning_rate=1e-3)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("=== Phase 1: frozen backbone ===")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs_head,
        batch_size=args.batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("=== Phase 2: fine-tune top MobileNet layers ===")
    unfreeze_top(model, n_layers=40)
    compile_model(model, learning_rate=1e-5)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs_ft,
        batch_size=args.batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_model(model, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

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
    metrics_path = ROOT / "models" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "val_loss")}, indent=2))

    compute_insights(TRAIN_DIR if TRAIN_DIR.exists() else args.data)
    print("insights cached")


if __name__ == "__main__":
    main()
