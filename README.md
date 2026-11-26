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

## Datasets

Two public fundus datasets, both graded on the same 5-level ICDR scale:
**`0 No_DR`, `1 Mild`, `2 Moderate`, `3 Severe`, `4 Proliferate_DR`**.

| Dataset | Images | Source | Role |
|---------|--------|--------|------|
| **APTOS 2019** | 3,662 (224×224) | HF mirror `bumbledeep/aptos` of [Diabetic Retinopathy 224×224 Gaussian Filtered (Kaggle)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered) | Primary dataset: training + the main held-out test set |
| **EyePACS** | 35,108 (~1024px) | Hugging Face [`ctmedtech/EYEPACS`](https://huggingface.co/datasets/ctmedtech/EYEPACS) (`EYEPACS.zip` + `trainLabels.csv`), a re-host of the labelled training set from Kaggle's 2015 [*Diabetic Retinopathy Detection*](https://www.kaggle.com/c/diabetic-retinopathy-detection) | Extra training data from different cameras/clinics + a second held-out test set |

The full split (85/15 per class for APTOS; patient-level 85/15 for EyePACS so a person's left and right eye never straddle train/test), the folder layout, and exactly which images were used for what are documented in **[`data/README.md`](data/README.md)**. Gaussian filtering is applied in `src/preprocessing.py` to every image at train and predict time.

**Citations**

- APTOS: Karthik, Maggie, and Sohier Dane. *APTOS 2019 Blindness Detection*. Kaggle.
- EyePACS (as requested by the dataset card): Kaggle and EyePACS. "Diabetic Retinopathy Detection." Kaggle Competition, 2015. https://www.kaggle.com/c/diabetic-retinopathy-detection. Accessed via Hugging Face: [`ctmedtech/EYEPACS`](https://huggingface.co/datasets/ctmedtech/EYEPACS).

**License note:** the EyePACS dataset card says use is restricted under Kaggle and EyePACS terms, and redistribution may be restricted. This repo does **not** include any EyePACS images (`data/raw/` and `data/splits/` are gitignored). Anyone reproducing the results downloads EyePACS themselves under those terms.

## Project structure

```
mlp-summative/
├── README.md
├── notebook/diabetic_retinopathy.ipynb
├── src/preprocessing.py | model.py | prediction.py | retrain.py | db.py | insights.py
├── api/main.py
├── ui/app.py
├── scripts/download_data.py | build_combined.py | train_model.py | evaluate.py | smoke_test.py
├── data/
│   ├── README.md            # dataset guide: what each folder is and how it was used
│   ├── raw/aptos/ | raw/eyepacs/        # source datasets (gitignored)
│   ├── splits/              # generated train/test splits (gitignored)
│   ├── samples/train/ | samples/test/   # small tracked demo sets (UI charts, Locust)
│   └── uploads/ + uploads.db            # images uploaded for retraining (runtime)
├── models/diabetic_retinopathy_mobilenet.h5 | metrics.json
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

# 1) Download APTOS into data/raw/aptos/ (+ small demo samples in data/samples/)
python scripts/download_data.py

# 2) EyePACS (manual): download EYEPACS.zip + trainLabels.csv from
#    https://huggingface.co/datasets/ctmedtech/EYEPACS, unzip the images into
#    data/raw/eyepacs/images/ and put trainLabels.csv in data/raw/eyepacs/

# 3) Build the train/test splits in data/splits/ (deterministic, seed 42)
python scripts/build_combined.py

# 4) Train the deployed model and score it on the held-out test sets
python scripts/train_model.py --data data/splits/mixed_train --augment \
    --epochs-head 10 --epochs-ft 20 --patience 6
python scripts/evaluate.py --model deployed=models/diabetic_retinopathy_mobilenet.h5 \
    --test aptos=data/splits/aptos_test --test eyepacs=data/splits/eyepacs_test

# 5) Run API + UI locally
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

## Model

**Deployed:** MobileNetV2 (ImageNet weights), fine-tuned on **all APTOS training images + 2,000 EyePACS images (400 per class)**, with train-time augmentation (random flips, full rotation, ±10 % zoom, brightness/contrast) — 10 frozen-backbone epochs + 20 fine-tune epochs, early-stopping patience 6. Trained on `data/splits/mixed_train` (5,112 images, 20 % held out for validation).

### Held-out test results

Scored on images **never used for training or validation** (from `models/metrics.json`):

| Test set | Images | Accuracy | Precision (macro) | Recall (macro) | **F1 (macro)** |
|----------|--------|----------|-------------------|----------------|----------------|
| APTOS test | 550 | **0.745** | 0.546 | 0.546 | **0.543** |
| EyePACS test (other cameras) | 300 | 0.420 | 0.354 | 0.420 | 0.369 |

Per-class F1:

| Test set | No_DR | Mild | Moderate | Severe | Proliferate_DR |
|----------|-------|------|----------|--------|----------------|
| APTOS | 0.96 | 0.57 | 0.62 | 0.19 | 0.38 |
| EyePACS | 0.32 | 0.45 | 0.00 | 0.42 | 0.66 |

Severe is the hardest grade: most missed Severe cases are predicted as Moderate (still a "refer to specialist" grade in DR screening), and only 1 of 29 APTOS Severe test images is predicted No_DR. Severe has just 29 APTOS test images, so its score is noisy.

### How it compares with the previous model

The previous model (APTOS only, 400 images per class, no augmentation) reported **F1 ≈ 0.60** in this README, but that number was inflated: its validation images were reused for testing, and many of the old `data/test` images were in its training set. Compared fairly on images neither model saw:

| | Previous model | **Deployed model** |
|---|---|---|
| Unseen APTOS test images (304 No_DR + Moderate) — accuracy | 0.793 | **0.852** |
| &nbsp;&nbsp;Moderate correctly graded | 45 / 93 | **54 / 93** |
| EyePACS test (300) — accuracy / macro F1 | 0.263 / 0.185 | **0.420 / 0.369** |
| EyePACS Severe F1 | 0.000 | **0.417** |

### Experiments

All models scored on the same held-out test sets (`scripts/evaluate.py`):

| Training data | Augment | Epochs (max) | APTOS F1 | EyePACS F1 |
|---------------|---------|--------------|----------|------------|
| APTOS, 400/class (previous recipe) | – | 10 | 0.513 | 0.177 |
| APTOS, all | – | 20 | 0.538 | 0.172 |
| APTOS, all | ✓ | 30 | 0.451 | 0.136 |
| APTOS + EyePACS, 400/class | – | 10 | 0.501 | 0.309 |
| APTOS all + EyePACS | – | 20 | 0.520 | 0.329 |
| **APTOS all + EyePACS (deployed)** | ✓ | 30 | **0.543** | **0.369** |

Findings: using all APTOS images helps; EyePACS only pays off with augmentation and a longer training budget, after which it matches the best APTOS-only model on APTOS while doubling performance on images from other cameras. Single training run per setting, so differences under ~0.02 F1 are within noise.

> `scripts/train_model.py` writes **validation** metrics to `models/metrics.json` when it trains the production model; the committed file holds the held-out **test** results above.

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
