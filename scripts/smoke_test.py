#!/usr/bin/env python3
"""Assert-based smoke checks for core pipeline pieces."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, MODEL_PATH, TRAIN_DIR
from src.db import init_db, list_uploads, save_upload
from src.preprocessing import preprocess_image
from src.insights import compute_insights


def main():
    sample_dirs = list(TRAIN_DIR.glob("*"))
    assert sample_dirs, "data/samples/train empty — run scripts/download_data.py"
    sample = next(TRAIN_DIR.joinpath(CLASS_NAMES[0]).glob("*"))
    tensor = preprocess_image(sample)
    assert tensor.shape == (224, 224, 3), tensor.shape
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0

    init_db()
    raw = sample.read_bytes()
    row = save_upload(sample.name, 0, raw)
    assert Path(row["path"]).exists()
    assert any(r["id"] == row["id"] for r in list_uploads())

    insights = compute_insights(TRAIN_DIR, max_per_class=5)
    assert "class_distribution" in insights
    assert len(insights["interpretations"]) >= 3

    if MODEL_PATH.exists():
        from src.prediction import clear_model_cache, predict_image

        clear_model_cache()
        pred = predict_image(sample)
        assert pred["class_name"] in CLASS_NAMES.values()
        assert 0.0 <= pred["confidence"] <= 1.0
        print("prediction ok:", pred["class_name"], pred["confidence"])
    else:
        print("model missing — skip predict check")

    print("smoke_test passed")


if __name__ == "__main__":
    main()
