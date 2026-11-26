# Diabetic Retinopathy MLOps Pipeline

End-to-end ML cycle for **5-class diabetic retinopathy grading** from retinal fundus images (non-tabular), with FastAPI serving, Streamlit UI, SQLite-backed upload → preprocess → retrain, Docker scaling, and Locust load tests.

Bridges the Intro-to-ML summative (diabetic patient outcomes) into an **image** clinical screening use case.

## Video demo

> YouTube link: _[add after recording — camera on, show predict + retrain]_

## Live URL

Deployed on AWS EC2 (eu-north-1) with the same Docker Compose stack:

- **UI (Streamlit):** http://51.20.133.231:8501
- **API (via nginx):** http://51.20.133.231:8000 — Swagger at http://51.20.133.231:8000/docs

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

## Deploy on AWS (EC2 + Docker Compose)

The whole Compose stack (api + nginx + ui) runs on a single EC2 instance, so the SQLite DB, uploads, and retrained models persist on the instance's EBS volume.

| Piece | Setting |
|-------|---------|
| Instance | `c7i-flex.large` (2 vCPU, 4 GB) + 4 GB swap, Amazon Linux 2023 |
| Disk | 30 GB gp3, encrypted |
| Network | Default VPC, public IP; security group opens **8000** (API) and **8501** (UI) only — no SSH |

Bootstrap: [`scripts/ec2-user-data.sh`](scripts/ec2-user-data.sh) is passed as EC2 user data. It adds swap, installs Docker + Compose/buildx, clones this repo to `/opt/app`, writes a `docker-compose.override.yml` with `restart: unless-stopped`, then runs `docker compose build && docker compose up -d`. First boot takes ~5 min (TensorFlow image build); log at `/var/log/app-bootstrap.log`.

Run:

```bash
# shell on the instance (no SSH key needed)
aws ssm start-session --target <instance-id> --region eu-north-1

# on the instance
cd /opt/app
sudo git pull && sudo docker compose up -d --build   # deploy latest code
sudo docker compose up -d --scale api=2              # scale API replicas behind nginx
sudo docker compose logs -f api
```

Notes:

- `/upload` and `/retrain` are unauthenticated and publicly reachable; restrict the security group to known IPs when not demoing.
- Stop the instance between demos to save cost (~$0.08/hr running); data on the EBS volume is kept, but the public IP changes.

## API quick reference

- `GET /health` — uptime + model status  
- `POST /predict` — multipart `file`  
- `POST /upload` — multipart `files` + form `labels` (`0,1,2,...`)  
- `POST /retrain` — JSON `{ "epochs": 3 }`  
- `GET /insights` — chart data + interpretations  

Swagger UI: `http://localhost:8000/docs` (local) · http://51.20.133.231:8000/docs (AWS)
