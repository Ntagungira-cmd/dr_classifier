# Diabetic Retinopathy MLOps Pipeline

End-to-end ML cycle for **5-class diabetic retinopathy grading** from retinal fundus images (non-tabular), with FastAPI serving, Streamlit UI, SQLite-backed upload → preprocess → retrain, Docker scaling, and Locust load tests.

Bridges the Intro-to-ML summative (diabetic patient outcomes) into an **image** clinical screening use case.

## Video demo

> YouTube link: _[add after recording — camera on, show predict + retrain]_

## Live URL

> Render API: _[add after deploy]_  
> Local UI: `http://localhost:8501`

## Dataset

Primary reference (assignment choice after survey of newer PhysioNet/IEEE options):

- [Diabetic Retinopathy 224×224 Gaussian Filtered (Kaggle)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)

Download path used in this repo (open HF mirror of APTOS 224×224): `bumbledeep/aptos`. Gaussian filtering is applied in `src/preprocessing.py` to match the Kaggle variant.

**Labels (ICDR):** `0 No_DR`, `1 Mild`, `2 Moderate`, `3 Severe`, `4 Proliferate_DR`

Citation: Karthik, Maggie, and Sohier Dane. APTOS 2019 Blindness Detection. Kaggle.

## Project structure

```
mlp-summative/
├── README.md
├── notebook/diabetic_retinopathy.ipynb
├── src/preprocessing.py | model.py | prediction.py | retrain.py | db.py | insights.py
├── api/main.py
├── ui/app.py
├── data/train/ | test/ | uploads/
├── models/diabetic_retinopathy_mobilenet.h5
├── locust/locustfile.py
├── docker-compose.yml
└── Dockerfile
```

## Setup

```bash
cd mlp-summative
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Download APTOS 224x224 + create train/test sample
python scripts/download_data.py

# 2) Train MobileNetV2 (uses data/full if present)
python scripts/train_model.py --max-per-class 200

# 3) Run API + UI locally
uvicorn api.main:app --reload --port 8000
# other terminal
API_URL=http://localhost:8000 streamlit run ui/app.py
```

### Docker (API + UI + nginx load balancer)

```bash
docker compose build
docker compose up --scale api=1
# UI http://localhost:8501  API via nginx http://localhost:8000
```

Scale API replicas for Locust comparison:

```bash
docker compose up -d --scale api=1 nginx
# run locust ...
docker compose up -d --scale api=2 nginx
docker compose up -d --scale api=3 nginx
```

## Features (assignment checklist)

| Requirement | Where |
|-------------|--------|
| Predict one fundus image | UI **Predict** tab / `POST /predict` |
| Visualizations (≥3 features + story) | UI **Visualizations** — class distribution, brightness, green channel (+ sharpness) |
| Bulk upload for retrain | UI **Upload data** / `POST /upload` → SQLite |
| Trigger retrain button | UI **Retrain** / `POST /retrain` (fine-tunes from custom `.h5`) |
| Model uptime | UI header metrics from `GET /health` |
| Notebook eval ≥4 metrics | `notebook/diabetic_retinopathy.ipynb` |
| Locust flood + multi-container latency | `locust/` + table below |

### Retrain trigger chain

1. Upload images → saved to SQLite + `data/uploads/`
2. Preprocess (resize, Gaussian, normalize)
3. Fine-tune starting from **your** saved MobileNetV2 checkpoint as pretrained weights

## Flood request simulation (Locust)

```bash
mkdir -p locust_results
# with stack up:
docker compose --profile load run --rm locust
# or locally:
locust -f locust/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 60s --csv locust_results/out
```

### Results

Local baseline (1 API process, Locust headless, 10 users, 20s):

| API containers | Users | RPS | Median latency (ms) | p95 latency (ms) | Failures |
|----------------|-------|-----|---------------------|------------------|----------|
| 1 (local uvicorn) | 10 | ~25 | 49 | 110 | 0 / 450 |
| 2 (docker compose `--scale api=2`) | 20 | _run locally_ | _run_ | _run_ | _run_ |
| 3 (docker compose `--scale api=3`) | 20 | _run locally_ | _run_ | _run_ | _run_ |

CSV outputs land in `locust_results/`. Re-run the scale matrix on your machine after `docker compose up --scale api=N`.

### Offline model metrics (validation)

From `models/metrics.json` (MobileNetV2, APTOS 224 subset): Accuracy ≈ **0.63**, Precision (macro) ≈ **0.61**, Recall (macro) ≈ **0.60**, F1 (macro) ≈ **0.60**, Loss ≈ **0.95**. No_DR F1 ≈ **0.92** (use a No_DR sample for a clear demo prediction).

## Deploy on Render (free)

1. Push this repo to GitHub
2. New Web Service → Docker → select repo
3. Or use `render.yaml` Blueprint
4. Ensure `models/diabetic_retinopathy_mobilenet.h5` is in the image or mounted
5. Free tier spins down after ~15 min idle; first request may take ~1 min to wake

## API quick reference

- `GET /health` — uptime + model status  
- `POST /predict` — multipart `file`  
- `POST /upload` — multipart `files` + form `labels` (`0,1,2,...`)  
- `POST /retrain` — JSON `{ "epochs": 3 }`  
- `GET /insights` — chart data + interpretations  

Swagger UI: `http://localhost:8000/docs`
