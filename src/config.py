from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# Source datasets as downloaded (gitignored) — see data/README.md
RAW_DIR = DATA_DIR / "raw"
FULL_DIR = RAW_DIR / "aptos"  # APTOS 2019, one folder per class
EYEPACS_DIR = RAW_DIR / "eyepacs"  # images/ + trainLabels.csv
# Generated train/test splits for model training (gitignored)
SPLITS_DIR = DATA_DIR / "splits"
# Small tracked demo samples (UI charts, Locust, smoke test) — NOT a held-out test set
SAMPLES_DIR = DATA_DIR / "samples"
TRAIN_DIR = SAMPLES_DIR / "train"
TEST_DIR = SAMPLES_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "uploads.db"
MODEL_PATH = MODELS_DIR / "diabetic_retinopathy_mobilenet.h5"
INSIGHTS_CACHE = DATA_DIR / "insights.json"

CLASS_NAMES = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}
CLASS_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
IMG_SIZE = 224
NUM_CLASSES = 5
SEED = 42
