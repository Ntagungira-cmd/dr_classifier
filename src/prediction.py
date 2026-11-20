"""Single-image prediction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.config import CLASS_NAMES, MODEL_PATH
from src.model import load_model
from src.preprocessing import preprocess_image


@lru_cache(maxsize=1)
def _cached_model(mtime: float, path: str):
    return load_model(Path(path))


def get_model(path: Optional[Path] = None):
    path = Path(path or MODEL_PATH)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return _cached_model(mtime, str(path))


def clear_model_cache() -> None:
    _cached_model.cache_clear()


def predict_image(
    source: Union[str, Path, bytes],
    *,
    model=None,
    model_path: Optional[Path] = None,
) -> dict:
    model = model or get_model(model_path)
    tensor = preprocess_image(source)
    batch = np.expand_dims(tensor, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    return {
        "class_id": pred_id,
        "class_name": CLASS_NAMES[pred_id],
        "confidence": float(probs[pred_id]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }
