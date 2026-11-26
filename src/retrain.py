"""Retrain trigger: preprocess DB uploads, fine-tune from custom checkpoint."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.config import CLASS_NAMES, MODEL_PATH, SEED
from src.db import (
    finish_retrain_run,
    list_uploads,
    mark_used,
    start_retrain_run,
)
from src.model import compile_model, load_model, save_model, unfreeze_top
from src.prediction import clear_model_cache
from src.preprocessing import preprocess_image


VAL_FRACTION = 0.25


def _load_xy_from_uploads(rows) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Load uploads; skip missing/unreadable files instead of failing the whole run."""
    xs, ys, ids, skipped = [], [], [], []
    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            skipped.append(f"{row['filename']} (file missing)")
            continue
        try:
            xs.append(preprocess_image(path))
        except Exception:
            skipped.append(f"{row['filename']} (unreadable image)")
            continue
        ys.append(int(row["label"]))
        ids.append(int(row["id"]))
    if not xs:
        raise ValueError("No valid uploaded images found for retraining")
    return np.stack(xs), np.array(ys, dtype=np.int32), ids, skipped


def _min_total_for_split(n_classes: int, val_fraction: float = VAL_FRACTION) -> int:
    """Smallest dataset size whose train and validation parts each fit every class."""
    n = 2 * n_classes
    while True:
        n_val = math.ceil(val_fraction * n)
        if n_val >= n_classes and n - n_val >= n_classes:
            return n
        n += 1


def check_trainable(y: np.ndarray, val_fraction: float = VAL_FRACTION) -> None:
    """
    Fail fast with an actionable message when the stratified split would crash.

    Mirrors run_retrain: with >= 4 images and more than one label, data is split
    into train/validation stratified by label, which needs >= 2 images per label
    and enough images for both parts to contain every label.
    """
    counts = Counter(int(v) for v in y)
    n, k = len(y), len(counts)
    if n < 4 or k < 2:
        return

    too_few = [f"{CLASS_NAMES.get(c, c)} (label {c}): {m}" for c, m in sorted(counts.items()) if m < 2]
    if too_few:
        raise ValueError(
            "Each label used for retraining needs at least 2 images. "
            f"Add more images for: {', '.join(too_few)}."
        )

    need = _min_total_for_split(k, val_fraction)
    if n < need:
        summary = ", ".join(f"{CLASS_NAMES.get(c, c)}: {m}" for c, m in sorted(counts.items()))
        raise ValueError(
            f"{n} images across {k} labels is too few: {int(val_fraction * 100)}% are held out "
            f"for validation and that part must include every label. Upload at least "
            f"{need - n} more (>= {need} total). Current counts: {summary}."
        )


def run_retrain(
    *,
    model_path: Path = MODEL_PATH,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    min_images: int = 2,
) -> dict:
    """
    Trigger chain:
    1) Read uploads saved in SQLite
    2) Preprocess
    3) Fine-tune starting from the custom saved model as pretrained weights
    """
    rows = list_uploads(unused_only=False)
    if len(rows) < min_images:
        raise ValueError(
            f"Need at least {min_images} uploaded labeled images; found {len(rows)}"
        )

    x, y, ids, skipped = _load_xy_from_uploads(rows)
    if len(ids) < min_images:
        raise ValueError(
            f"Need at least {min_images} readable labeled images; found {len(ids)}"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
        )
    check_trainable(y)
    run_id = start_retrain_run(len(ids))

    try:
        # Load custom checkpoint as pretrained starting point
        model = load_model(model_path)
        unfreeze_top(model, n_layers=30)
        compile_model(model, learning_rate=learning_rate)

        if len(x) >= 4:
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=VAL_FRACTION, random_state=SEED, stratify=y if len(set(y)) > 1 else None
            )
        else:
            x_train, y_train = x, y
            x_val, y_val = x, y

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=min(batch_size, len(x_train)),
            callbacks=callbacks,
            verbose=1,
        )
        # Keep previous weights so a bad retrain can be rolled back manually
        if model_path.exists():
            backup = model_path.with_suffix(model_path.suffix + ".bak")
            backup.write_bytes(model_path.read_bytes())
        save_model(model, model_path)
        clear_model_cache()
        mark_used(ids)

        metrics = {
            "n_images": len(ids),
            "epochs_ran": len(history.history.get("loss", [])),
            "final_accuracy": float(history.history.get("accuracy", [0])[-1]),
            "final_val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
            "final_loss": float(history.history.get("loss", [0])[-1]),
            "class_names": CLASS_NAMES,
            "skipped": skipped,
        }
        finish_retrain_run(run_id, "success", json.dumps(metrics))
        return {"run_id": run_id, "status": "success", "metrics": metrics}
    except Exception as exc:
        finish_retrain_run(run_id, "failed", json.dumps({"error": str(exc)}))
        raise
