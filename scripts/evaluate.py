#!/usr/bin/env python3
"""Score one or more models on held-out test folders (one subfolder per class).

Example:
  python scripts/evaluate.py \
      --model aptos_only=models/experiments/aptos_only.h5 \
      --model mixed=models/experiments/mixed.h5 \
      --test aptos=data/combined/aptos_test --test eyepacs=data/combined/eyepacs_test \
      --json-out models/experiments/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES  # noqa: E402
from src.model import load_model  # noqa: E402
from src.preprocessing import preprocess_image  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg"}


def load_test(root: Path):
    xs, ys = [], []
    for label, name in CLASS_NAMES.items():
        for p in sorted((root / name).glob("*")):
            if p.suffix.lower() in EXTS:
                xs.append(preprocess_image(p))
                ys.append(label)
    if not xs:
        raise SystemExit(f"no images under {root}")
    return np.stack(xs), np.array(ys)


def named(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not path:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    return name, Path(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=named, action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--test", type=named, action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    tests = {name: load_test(path) for name, path in args.test}
    labels = list(CLASS_NAMES)
    names = [CLASS_NAMES[i] for i in labels]
    results = {}
    for mname, mpath in args.model:
        model = load_model(mpath)
        results[mname] = {}
        for tname, (x, y) in tests.items():
            pred = model.predict(x, batch_size=64, verbose=0).argmax(axis=1)
            rep = classification_report(y, pred, labels=labels, target_names=names, zero_division=0, output_dict=True)
            results[mname][tname] = {
                "n": int(len(y)),
                "accuracy": float(accuracy_score(y, pred)),
                "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
                "f1_per_class": {n: float(rep[n]["f1-score"]) for n in names},
                "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            }

    for tname in tests:
        print(f"\n=== {tname} test (n={len(tests[tname][1])}) ===")
        print(f"{'model':14}{'acc':>7}{'F1mac':>7}" + "".join(f"{n[:8]:>10}" for n in names))
        for mname in results:
            r = results[mname][tname]
            print(f"{mname:14}{r['accuracy']:7.3f}{r['f1_macro']:7.3f}"
                  + "".join(f"{r['f1_per_class'][n]:10.3f}" for n in names))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
