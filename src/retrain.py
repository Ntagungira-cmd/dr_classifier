"""Retrain trigger: preprocess DB uploads, fine-tune from custom checkpoint."""

from __future__ import annotations

import json
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


def _load_xy_from_uploads(rows) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs, ys, ids = [], [], []
    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            continue
        xs.append(preprocess_image(path))
        ys.append(int(row["label"]))
        ids.append(int(row["id"]))
    if not xs:
        raise ValueError("No valid uploaded images found for retraining")
    return np.stack(xs), np.array(ys, dtype=np.int32), ids


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

    x, y, ids = _load_xy_from_uploads(rows)
    run_id = start_retrain_run(len(ids))

    try:
        # Load custom checkpoint as pretrained starting point
        model = load_model(model_path)
        unfreeze_top(model, n_layers=30)
        compile_model(model, learning_rate=learning_rate)

        if len(x) >= 4:
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=0.25, random_state=SEED, stratify=y if len(set(y)) > 1 else None
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
        }
        finish_retrain_run(run_id, "success", json.dumps(metrics))
        return {"run_id": run_id, "status": "success", "metrics": metrics}
    except Exception as exc:
        finish_retrain_run(run_id, "failed", json.dumps({"error": str(exc)}))
        raise
