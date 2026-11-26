#!/usr/bin/env python3
"""Build a clean APTOS + EyePACS train/test split under data/splits/.

Layout (one folder per class inside each split):
  aptos_train/    APTOS train images (symlinks into data/raw/aptos); all by default
  aptos_test/     APTOS held-out test set: never used for training
  eyepacs_train/  EyePACS sample, resized to IMG_SIZE
  eyepacs_test/   EyePACS held-out test set (patients disjoint from eyepacs_train)
  mixed_train/    aptos_train + eyepacs_train (symlinks), for the combined model
  split_manifest.csv  source, image, label, split for every file

EyePACS is split by patient (e.g. 10_left / 10_right are one patient) so the two
eyes of one person never end up on different sides of the train/test split.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, EYEPACS_DIR, FULL_DIR, IMG_SIZE, SEED, SPLITS_DIR  # noqa: E402

OUT_DIR = SPLITS_DIR


def link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src.resolve(), dest.parent), dest)


def split_aptos(test_frac: float, train_cap: int, rng: random.Random):
    train, test = [], []
    for label, name in CLASS_NAMES.items():
        files = sorted(p for p in (FULL_DIR / name).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        rng.shuffle(files)
        n_test = round(len(files) * test_frac)
        test += [(f, label) for f in files[:n_test]]
        rest = files[n_test:]
        train += [(f, label) for f in (rest[:train_cap] if train_cap else rest)]
    return train, test


def split_eyepacs(labels_csv: Path, img_dir: Path, test_frac: float,
                  train_cap: int, test_cap: int, rng: random.Random):
    with labels_csv.open() as fh:
        labels = {r["image"]: int(r["level"]) for r in csv.DictReader(fh)}
    by_patient = defaultdict(list)
    for img, label in labels.items():
        path = img_dir / f"{img}.jpeg"
        if path.exists():
            by_patient[img.split("_")[0]].append((path, label))

    patients = sorted(by_patient)
    rng.shuffle(patients)
    n_test = round(len(patients) * test_frac)
    pools = {"test": patients[:n_test], "train": patients[n_test:]}
    caps = {"train": train_cap, "test": test_cap}

    out = {}
    for split, pats in pools.items():
        per_class = defaultdict(list)
        for p in pats:
            for path, label in by_patient[p]:
                per_class[label].append(path)
        chosen = []
        for label in CLASS_NAMES:
            files = per_class[label]
            rng.shuffle(files)
            chosen += [(f, label) for f in files[: caps[split]]]
        out[split] = chosen
    assert not ({f.stem.split("_")[0] for f, _ in out["train"]}
                & {f.stem.split("_")[0] for f, _ in out["test"]}), "patient leak"
    return out["train"], out["test"]


def write_resized(src: Path, dest: Path) -> None:
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"unreadable: {src}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aptos-test-frac", type=float, default=0.15)
    ap.add_argument("--aptos-train-cap", type=int, default=0, help="per class; 0 = all (deployed model)")
    ap.add_argument("--eyepacs-test-frac", type=float, default=0.15, help="fraction of patients held out")
    ap.add_argument("--eyepacs-train-cap", type=int, default=400, help="per class")
    ap.add_argument("--eyepacs-test-cap", type=int, default=60, help="per class")
    ap.add_argument("--dry-run", action="store_true", help="print counts only")
    args = ap.parse_args()

    labels_csv = EYEPACS_DIR / "trainLabels.csv"
    img_dir = EYEPACS_DIR / "images"
    for p in (FULL_DIR / "No_DR", labels_csv, img_dir):
        if not p.exists():
            sys.exit(f"missing {p}")

    aptos_train, aptos_test = split_aptos(args.aptos_test_frac, args.aptos_train_cap, random.Random(SEED))
    eye_train, eye_test = split_eyepacs(labels_csv, img_dir, args.eyepacs_test_frac,
                                        args.eyepacs_train_cap, args.eyepacs_test_cap, random.Random(SEED))
    splits = {"aptos_train": aptos_train, "aptos_test": aptos_test,
              "eyepacs_train": eye_train, "eyepacs_test": eye_test}

    names = [CLASS_NAMES[i] for i in CLASS_NAMES]
    print(f"{'split':15}" + "".join(f"{n:>15}" for n in names) + f"{'total':>8}")
    for split, items in splits.items():
        c = Counter(label for _, label in items)
        print(f"{split:15}" + "".join(f"{c[i]:>15}" for i in CLASS_NAMES) + f"{len(items):>8}")
    mixed = Counter(label for _, label in aptos_train + eye_train)
    print(f"{'mixed_train':15}" + "".join(f"{mixed[i]:>15}" for i in CLASS_NAMES) + f"{sum(mixed.values()):>8}")
    if args.dry_run:
        return

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    rows = []
    for split, items in splits.items():
        for i, (src, label) in enumerate(items):
            cls = CLASS_NAMES[label]
            if split.startswith("aptos"):
                dest = OUT_DIR / split / cls / src.name
                link(src, dest)
            else:
                dest = OUT_DIR / split / cls / f"{src.stem}.png"
                write_resized(src, dest)
            rows.append({"source": split.split("_")[0], "image": src.name, "label": label, "split": split})
            if split.startswith("eyepacs") and (i + 1) % 250 == 0:
                print(f"  {split}: {i + 1}/{len(items)} resized")
    for split in ("aptos_train", "eyepacs_train"):
        for f in (OUT_DIR / split).rglob("*"):
            if f.is_file() or f.is_symlink():
                link(f, OUT_DIR / "mixed_train" / f.parent.name / f.name)

    with (OUT_DIR / "split_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "image", "label", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
