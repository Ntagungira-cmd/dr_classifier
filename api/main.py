"""FastAPI service: predict, upload, retrain, health, insights."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, MODEL_PATH
from src.db import init_db, list_uploads, save_upload, upload_counts_by_label
from src.insights import load_insights
from src.prediction import clear_model_cache, predict_image
from src.preprocessing import validate_image_bytes
from src.retrain import backup_path, restore_backup, run_retrain

STARTED_AT = datetime.now(timezone.utc)
app = FastAPI(
    title="Diabetic Retinopathy API",
    description="Predict DR grade from fundus images; upload + retrain pipeline.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict
    latency_ms: float


class UploadItem(BaseModel):
    id: int
    filename: str
    label: int
    path: str


class RetrainRequest(BaseModel):
    epochs: int = Field(3, ge=1, le=20)
    min_images: int = Field(2, ge=1)


@app.on_event("startup")
def _startup():
    init_db()
    # Warm model if present
    if MODEL_PATH.exists():
        try:
            from src.prediction import get_model

            get_model()
        except Exception as exc:
            print(f"model warm failed: {exc}")


@app.get("/health")
def health():
    uptime = (datetime.now(timezone.utc) - STARTED_AT).total_seconds()
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "backup_available": backup_path().exists(),
        "uptime_seconds": uptime,
        "started_at": STARTED_AT.isoformat(),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }


@app.get("/classes")
def classes():
    return CLASS_NAMES


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model file missing — train first")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    t0 = time.perf_counter()
    try:
        result = predict_image(data)
    except Exception as exc:
        raise HTTPException(400, f"Prediction failed: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(**result, latency_ms=latency_ms)


@app.post("/upload", response_model=List[UploadItem])
async def upload(
    files: List[UploadFile] = File(...),
    labels: str = Form(..., description="Comma-separated labels 0-4, one per file"),
):
    try:
        label_list = [int(x.strip()) for x in labels.split(",") if x.strip() != ""]
    except ValueError:
        raise HTTPException(400, f"Labels must be comma-separated numbers 0-4; got {labels!r}")
    if len(label_list) != len(files):
        raise HTTPException(
            400,
            f"labels count ({len(label_list)}) must match files ({len(files)})",
        )
    # Validate the whole batch before saving anything
    batch = []
    for f, label in zip(files, label_list):
        name = f.filename or "image.jpg"
        if label not in CLASS_NAMES:
            raise HTTPException(400, f"{name}: invalid label {label}; expected 0-4")
        data = await f.read()
        try:
            validate_image_bytes(data)
        except ValueError as exc:
            raise HTTPException(400, f"{name}: {exc}. Nothing was saved.") from exc
        batch.append((name, label, data))
    return [UploadItem(**save_upload(name, label, data)) for name, label, data in batch]


@app.get("/uploads")
def uploads():
    rows = list_uploads()
    return [
        {
            "id": r["id"],
            "filename": r["filename"],
            "label": r["label"],
            "path": r["path"],
            "created_at": r["created_at"],
            "used_in_retrain": r["used_in_retrain"],
        }
        for r in rows
    ]


@app.post("/retrain")
def retrain(body: RetrainRequest = RetrainRequest()):
    try:
        result = run_retrain(epochs=body.epochs, min_images=body.min_images)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Retrain failed: {exc}") from exc
    clear_model_cache()
    return result


@app.post("/reset-model")
def reset_model():
    """Undo the last retrain: delete the current model and restore the .h5.bak backup."""
    try:
        return restore_backup()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/insights")
def insights():
    try:
        return load_insights()
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/upload-stats")
def upload_stats():
    return {
        "by_label": [
            {"label": lab, "class_name": CLASS_NAMES[lab], "count": n}
            for lab, n in upload_counts_by_label()
        ]
    }
