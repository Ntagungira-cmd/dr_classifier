"""Dataset feature insights for the UI (≥3 interpreted features)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from src.config import CLASS_NAMES, INSIGHTS_CACHE, TRAIN_DIR
from src.preprocessing import image_stats


def _iter_labeled_images(root: Path):
    for class_id, name in CLASS_NAMES.items():
        folder = root / name
        if not folder.is_dir():
            # also accept numeric folder names
            folder = root / str(class_id)
        if not folder.is_dir():
            continue
        for p in folder.glob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                yield class_id, name, p


def compute_insights(
    data_dir: Path = TRAIN_DIR,
    *,
    max_per_class: int = 80,
    cache_path: Optional[Path] = INSIGHTS_CACHE,
) -> dict:
    counts = Counter()
    brightness = defaultdict(list)
    green = defaultdict(list)
    sharpness = defaultdict(list)

    per_class_seen = Counter()
    for class_id, name, path in _iter_labeled_images(data_dir):
        if per_class_seen[class_id] >= max_per_class:
            continue
        per_class_seen[class_id] += 1
        counts[name] += 1
        stats = image_stats(path)
        brightness[name].append(stats["brightness"])
        green[name].append(stats["green_mean"])
        sharpness[name].append(stats["sharpness"])

    def avg(d):
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in d.items()}

    result = {
        "class_distribution": dict(counts),
        "mean_brightness_by_class": avg(brightness),
        "mean_green_channel_by_class": avg(green),
        "mean_sharpness_by_class": avg(sharpness),
        "interpretations": {
            "class_distribution": (
                "Class counts show screening skew: No_DR usually dominates. "
                "That imbalance is why we use class weights / careful metrics "
                "(not accuracy alone) when judging model quality."
            ),
            "mean_brightness_by_class": (
                "Average luminance by grade reflects acquisition conditions "
                "(underexposure / overexposure). Systematic brightness shifts "
                "can mimic pathology cues, so Gaussian filtering + normalization "
                "stabilize training across clinics."
            ),
            "mean_green_channel_by_class": (
                "The green channel carries most vessel/lesion contrast in fundus "
                "photos. Higher green means by severity often track hemorrhages "
                "and exudates becoming more visible — a clinically meaningful "
                "image feature for DR grading."
            ),
            "mean_sharpness_by_class": (
                "Laplacian sharpness summarizes focus quality. Soft/blurry frames "
                "are harder to grade; sharpness gaps across classes warn that "
                "the model may be learning camera quality as a shortcut."
            ),
        },
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2))
    return result


def load_insights(cache_path: Path = INSIGHTS_CACHE, data_dir: Path = TRAIN_DIR) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return compute_insights(data_dir, cache_path=cache_path)
