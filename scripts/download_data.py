#!/usr/bin/env python3
"""Download APTOS 224x224 fundus images (HF mirror of the Kaggle APTOS resize).

Primary dataset referenced in README:
https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered

We apply Gaussian filtering in src.preprocessing to match that variant.
Fallback HF dataset: bumbledeep/aptos (3662 images, 224x224, labels 0-4).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, FULL_DIR, SEED, TEST_DIR, TRAIN_DIR


def download_hf(repo: str = "bumbledeep/aptos", out: Path = FULL_DIR) -> Path:
    from datasets import load_dataset

    out.mkdir(parents=True, exist_ok=True)
    for name in CLASS_NAMES.values():
        (out / name).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo, split="train")
    for i, row in enumerate(ds):
        label = int(row["label_code"])
        name = CLASS_NAMES[label]
        img = row["image"]
        dest = out / name / f"{i:05d}.png"
        img.save(dest)
        if (i + 1) % 500 == 0:
            print(f"saved {i + 1} images...")
    print(f"Downloaded {len(ds)} images into {out}")
    return out


def split_sample(
    full: Path = FULL_DIR,
    *,
    train_per_class: int = 40,
    test_per_class: int = 10,
) -> None:
    random.seed(SEED)
    for d in (TRAIN_DIR, TEST_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for class_id, name in CLASS_NAMES.items():
        src = full / name
        files = sorted(src.glob("*"))
        random.shuffle(files)
        n_train = min(train_per_class, max(0, len(files) - test_per_class))
        n_test = min(test_per_class, len(files) - n_train)
        train_files = files[:n_train]
        test_files = files[n_train : n_train + n_test]
        (TRAIN_DIR / name).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / name).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(f, TRAIN_DIR / name / f.name)
        for f in test_files:
            shutil.copy2(f, TEST_DIR / name / f.name)
        print(f"{name}: train={len(train_files)} test={len(test_files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="bumbledeep/aptos")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--test-per-class", type=int, default=10)
    args = parser.parse_args()

    if not args.skip_download:
        download_hf(args.repo, FULL_DIR)
    split_sample(
        FULL_DIR,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    print(f"Sample ready under {TRAIN_DIR} and {TEST_DIR}")


if __name__ == "__main__":
    main()
