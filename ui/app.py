"""Streamlit UI: uptime, insights, predict, upload, retrain."""

from __future__ import annotations

import os
from typing import List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

CLASS_LABELS = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}

st.set_page_config(
    page_title="DR Screening MLOps",
    page_icon="👁️",
    layout="wide",
)


class APIError(Exception):
    """API call failed; message is the API's own `detail` when it sent one."""


def _check(r: requests.Response):
    if r.ok:
        return r.json()
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = None
    raise APIError(detail if isinstance(detail, str) else f"{r.status_code} {r.reason}: {r.text[:300]}")


def api_get(path: str, timeout: float = 10):
    return _check(requests.get(f"{API_URL}{path}", timeout=timeout))


def api_post(path: str, **kwargs):
    return _check(requests.post(f"{API_URL}{path}", timeout=kwargs.pop("timeout", 120), **kwargs))


st.title("Diabetic Retinopathy MLOps")
st.caption(f"API: `{API_URL}` — fundus image grading (APTOS-style 0–4)")

# --- Uptime ---
col_h1, col_h2, col_h3 = st.columns(3)
try:
    health = api_get("/health")
    col_h1.metric("API status", health.get("status", "?").upper())
    col_h2.metric("Uptime (s)", f"{health.get('uptime_seconds', 0):.0f}")
    col_h3.metric("Model loaded", "Yes" if health.get("model_loaded") else "No")
except Exception as exc:
    st.error(f"API unreachable: {exc}")
    st.stop()

tab_predict, tab_upload, tab_insights, tab_retrain = st.tabs(
    ["Predict", "Upload data", "Visualizations", "Retrain"]
)

with tab_predict:
    st.subheader("Single fundus image prediction")
    up = st.file_uploader("Upload one .jpg / .png", type=["jpg", "jpeg", "png"], key="pred")
    if up and st.button("Run prediction", type="primary"):
        with st.spinner("Predicting..."):
            files = {"file": (up.name, up.getvalue(), up.type or "image/jpeg")}
            try:
                result = api_post("/predict", files=files, timeout=60)
            except Exception as exc:
                st.error(exc)
            else:
                st.image(up.getvalue(), caption=up.name, width=280)
                st.success(
                    f"**{result['class_name']}** (id={result['class_id']}) "
                    f"— confidence {result['confidence']:.2%} "
                    f"— latency {result['latency_ms']:.0f} ms"
                )
                st.bar_chart(pd.Series(result["probabilities"], name="probability"))

with tab_upload:
    st.subheader("Bulk upload for retraining")
    st.write("Upload multiple images and assign a DR label (0–4) to each batch, or one label for all.")
    files = st.file_uploader(
        "Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="bulk",
    )
    mode = st.radio("Label mode", ["Same label for all", "Comma-separated labels"], horizontal=True)
    if mode == "Same label for all":
        label = st.selectbox(
            "Label",
            options=list(CLASS_LABELS.keys()),
            format_func=lambda i: f"{i} — {CLASS_LABELS[i]}",
        )
        labels_str = ",".join([str(label)] * len(files)) if files else ""
    else:
        labels_str = st.text_input("Labels (comma-separated, one per file)", "")
    if files and st.button("Save to database"):
        multipart = [("files", (f.name, f.getvalue(), f.type or "image/jpeg")) for f in files]
        try:
            saved = api_post("/upload", files=multipart, data={"labels": labels_str}, timeout=120)
            st.success(f"Stored {len(saved)} images in SQLite")
            st.json(saved)
        except Exception as exc:
            st.error(exc)

    try:
        stats = api_get("/upload-stats")
        if stats["by_label"]:
            st.write("Uploads in DB")
            st.dataframe(pd.DataFrame(stats["by_label"]))
    except Exception:
        pass

with tab_insights:
    st.subheader("Dataset feature visualizations")
    try:
        insights = api_get("/insights")
    except Exception as exc:
        st.warning(f"Insights unavailable: {exc}")
    else:
        dist = insights.get("class_distribution", {})
        if dist:
            fig = px.bar(
                x=list(dist.keys()),
                y=list(dist.values()),
                labels={"x": "DR grade", "y": "count"},
                title="1) Class distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(insights["interpretations"]["class_distribution"])

        bright = insights.get("mean_brightness_by_class", {})
        if bright:
            fig = px.bar(
                x=list(bright.keys()),
                y=list(bright.values()),
                labels={"x": "DR grade", "y": "mean luminance"},
                title="2) Mean brightness by class",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(insights["interpretations"]["mean_brightness_by_class"])

        green = insights.get("mean_green_channel_by_class", {})
        if green:
            fig = px.bar(
                x=list(green.keys()),
                y=list(green.values()),
                labels={"x": "DR grade", "y": "mean green channel"},
                title="3) Mean green-channel intensity by class",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(insights["interpretations"]["mean_green_channel_by_class"])

        sharp = insights.get("mean_sharpness_by_class", {})
        if sharp:
            fig = px.bar(
                x=list(sharp.keys()),
                y=list(sharp.values()),
                labels={"x": "DR grade", "y": "Laplacian variance"},
                title="4) Mean sharpness by class (bonus)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(insights["interpretations"]["mean_sharpness_by_class"])

with tab_retrain:
    st.subheader("Trigger retraining")
    st.write(
        "Uses uploaded images saved in the database, preprocesses them, "
        "and fine-tunes from the custom saved MobileNetV2 checkpoint."
    )
    epochs = st.slider("Epochs", 1, 10, 3)
    if st.button("Start retrain", type="primary"):
        with st.spinner("Retraining — this may take a few minutes..."):
            try:
                result = api_post(
                    "/retrain",
                    json={"epochs": epochs, "min_images": 2},
                    timeout=600,
                )
                st.success("Retrain finished")
                skipped = result.get("metrics", {}).get("skipped") or []
                if skipped:
                    st.warning("Skipped unreadable uploads: " + ", ".join(skipped))
                st.json(result)
            except Exception as exc:
                st.error(exc)
