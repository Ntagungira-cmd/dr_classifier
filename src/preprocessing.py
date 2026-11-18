"""Shared image preprocessing for train / predict / retrain."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from src.config import IMG_SIZE


def apply_gaussian(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Match the APTOS Gaussian-filtered Kaggle variant."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def load_image_rgb(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    if isinstance(source, np.ndarray):
        img = source
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img.astype(np.uint8)

    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    path = Path(source)
    bgr = cv2.imread(str(path))
    if bgr is None:
        # Pillow fallback for uncommon encodings
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(
    source: Union[str, Path, bytes, np.ndarray],
    *,
    size: int = IMG_SIZE,
    gaussian: bool = True,
) -> np.ndarray:
    """Return float32 tensor (size, size, 3) in [0, 1]."""
    img = load_image_rgb(source)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if gaussian:
        img = apply_gaussian(img)
    return (img.astype(np.float32) / 255.0)


def preprocess_batch(paths, **kwargs) -> np.ndarray:
    return np.stack([preprocess_image(p, **kwargs) for p in paths], axis=0)


def image_stats(source: Union[str, Path, bytes, np.ndarray]) -> dict:
    """Pixel-level features used for insights visualizations."""
    img = load_image_rgb(source)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Laplacian variance ≈ sharpness
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return {
        "brightness": float(gray.mean()),
        "green_mean": float(img[:, :, 1].mean()),
        "red_mean": float(img[:, :, 0].mean()),
        "blue_mean": float(img[:, :, 2].mean()),
        "sharpness": sharp,
        "contrast": float(gray.std()),
    }


def bytes_from_array(img: np.ndarray, ext: str = ".jpg") -> bytes:
    rgb = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise ValueError("Failed to encode image")
    return buf.tobytes()
