from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
FULL_DIR = DATA_DIR / "full"
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
