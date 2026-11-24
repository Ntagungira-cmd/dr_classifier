"""Flood /predict with a sample fundus image."""

from __future__ import annotations

import os
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CANDIDATES = [
    ROOT / "data" / "test" / "No_DR",
    ROOT / "data" / "train" / "No_DR",
    ROOT / "data" / "test" / "Mild",
    ROOT / "data" / "train" / "Mild",
]


def _sample_image() -> Path:
    override = os.getenv("LOCUST_IMAGE")
    if override and Path(override).exists():
        return Path(override)
    for folder in SAMPLE_CANDIDATES:
        if folder.is_dir():
            files = list(folder.glob("*"))
            if files:
                return files[0]
    raise FileNotFoundError("No sample image found under data/train or data/test")


SAMPLE = _sample_image()
IMAGE_BYTES = SAMPLE.read_bytes()


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        files = {"file": (SAMPLE.name, IMAGE_BYTES, "image/png")}
        with self.client.post("/predict", files=files, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}: {resp.text[:200]}")
