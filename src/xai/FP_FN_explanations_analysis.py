#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper outputs: aggregate SHAP comparisons (collapsed over time)
- FP vs TN  (within negatives)  [class-1 view]
- FN vs TP  (within positives)  [class-1 view]

Artifacts (saved under ./analysis_paper/):
- FP_vs_TN_collapsed.csv
- FN_vs_TP_collapsed.csv
- FP_vs_TN_top15.pdf
- FN_vs_TP_top15.pdf
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# -------------------------------
# Paths (adjust if needed)
# -------------------------------
MODEL_PATH   = '../saved_models/day_5_GTPU_non_resampled_model.h5'
FEATURES_CSV = "../features.csv"
LABELS_CSV   = "../labels.csv"

# -------------------------------
# Load model & data
# -------------------------------
print("TensorFlow:", tf.__version__)
# If you hit optimizer/loss deserialization issues, set compile=False
model = load_model(MODEL_PATH, compile=True)

X_flattened = pd.read_csv(FEATURES_CSV)  # shape: [N, 42] (3 time steps × 14 features)
y_true = pd.read_csv(LABELS_CSV).values.flatten().astype(int)

# -------------------------------
# Feature metadata
# -------------------------------
features = [
    'epre', 'pusch_snr', 'p_ue', 'ul_mcs', 'cqi', 'ul_bitrate',
    'dl_mcs', 'dl_retx', 'ul_tx', 'dl_tx', 'ul_retx', 'dl_bitrate',
    'dl_err', 'ul_err'
]
T = 3
flattened_feature_names = [f"{feat}_T{t}" for t in range(T) for feat in features]
assert X_flattened.shape[1] == len(flattened_feature_names), "Unexpected # feature columns."

# -------------------------------
# Predict
# -------------------------------
X = X_flattened.values.reshape(-1, T, len(features))
y_pred_prob = model.predict(X, verbose=0).ravel()       # P(class=1)
y_pred = (y_pred_prob > 0.5).astype(int)

# Buckets
fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]

print(f"Counts — TP:{len(tp_idx)}  TN:{len(tn_idx)}  FP:{len(fp_idx)}  FN:{len(fn_idx)}")

# -------------------------------
# SHAP (Kernel) – model-agnostic
# -------------------------------
def predict_fn(input_2d_flat):
    """input [B, 42] -> output [B, 2] (class0, class1) probabilities."""
    r = input_2d_flat.reshape(-1, T, len(features))
    p1 = model.predict(r, verbose=0).ravel()[:, None]
    p0 = 1.0 - p1
    return np.hstack((p0, p1))

# Small background for speed (tune if needed)
background = X_flattened.iloc[: min(100, len(X_flattened))].copy()
shap_explainer = shap.KernelExplainer(predict_fn, background)

# Robustly normalize SHAP to [n_samples, n_features] for class-1
def shap_to_3d(shap_vals):
    """Return ndarray [n, f, c] from SHAP outputs (handles list/tuple and 2D/3D arrays)."""
    if isinstance(shap_vals, (list, tuple)):
        arrs = [np.asarray(a) for a in shap_vals]
        arr = np.stack(arrs, axis=-1)  # (n,f,c)
    else:
        arr = np.asarray(shap_vals)
        if arr.ndim == 2:  # (n,f)
            arr = arr[..., None]       # -> (n,f,1)
        elif arr.ndim != 3:
            raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    return arr

def shap_class1_2d(shap_vals):
    """Always take class-1 SHAPs (Attack). If only 1 class returned, use that."""
    arr = shap_to_3d(shap_vals)  # (n,f,c)
    cls = 1 if arr.shape[-1] >= 2 else 0
    return arr[:, :, cls]        # -> (n,f)

# -------------------------------
# Utilities
# -------------------------------
def sample_indices(arr, max_n=200, seed=42):
    if len(arr) <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(arr, size=max_n, replace=False))

def mean_abs_shap(shap_2d):
    """Return Series over 42 flattened features (mean |SHAP|)."""
    return pd.Series(np.abs(shap_2d).mean(axis=0), index=flattened_feature_names)

def collapse_over_time(series_flat):
    """Sum T0+T1+T2 per base feature -> Series of length 14."""
    df = series_flat.reset_index()
    df.columns = ["flat_feature", "value"]
    df["base_feature"] = df["flat_feature"].str.replace(r"_T\d+$", "", regex=True)
    return df.groupby("base_feature")["value"].sum().sort_values(ascending=False)

def run_pair_and_save(name, idx_A, idx_B, out_dir):
    """
    Compute mean |SHAP| (class-1) for group A and B, collapse over time,
    and save CSV + Top15 delta bar plot.
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(idx_A) == 0 or len(idx_B) == 0:
        print(f"[{name}] One group empty; skipping.")
        return

    # Speed cap
    A = sample_indices(idx_A, 200)
    B = sample_indices(idx_B, 200)

    # Compute SHAP
    raw_A = shap_explainer.shap_values(X_flattened.iloc[A])
    raw_B = shap_explainer.shap_values(X_flattened.iloc[B])

    shap_A = shap_class1_2d(raw_A)  # (na, 42)
    shap_B = shap_class1_2d(raw_B)  # (nb, 42)

    # Mean |SHAP| over samples (42), then collapse to base features (14)
    ma_A_flat = mean_abs_shap(shap_A)
    ma_B_flat = mean_abs_shap(shap_B)
    ma_A = collapse_over_time(ma_A_flat)
    ma_B = collapse_over_time(ma_B_flat)

    # Align & delta
    comp = pd.concat([ma_A.rename(f"{name}_A_mean_abs"),
                      ma_B.rename(f"{name}_B_mean_abs")], axis=1).fillna(0.0)
    comp["delta_A_minus_B"] = comp.iloc[:, 0] - comp.iloc[:, 1]
    comp = comp.reindex(comp["delta_A_minus_B"].abs().sort_values(ascending=False).index)

    # Save CSV
    csv_path = os.path.join(out_dir, f"{name}_collapsed.csv")
    comp.to_csv(csv_path)

    # Plot Top-15 |Δ|
    top = comp.head(15)
    plt.figure(figsize=(8, 5))
    top["delta_A_minus_B"].iloc[::-1].plot(kind="barh")
    plt.xlabel("Δ mean |SHAP| (A − B) summed over time")
    plt.ylabel("Feature")
    plt.title(f"{name}: Top 15 |Δ| (class-1 SHAP, collapsed)")
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, f"{name}_top15.pdf")
    plt.savefig(pdf_path, bbox_inches="tight"); plt.close()

    print(f"[{name}] Wrote:\n  {csv_path}\n  {pdf_path}")

# -------------------------------
# Run (class-1 view only; paper-ready)
# -------------------------------
OUT = "analysis_paper"
os.makedirs(OUT, exist_ok=True)

# Pair 1: FP (A) vs TN (B) within negatives
run_pair_and_save(name="FP_vs_TN", idx_A=fp_idx, idx_B=tn_idx, out_dir=OUT)

# Pair 2: FN (A) vs TP (B) within positives
run_pair_and_save(name="FN_vs_TP", idx_A=fn_idx, idx_B=tp_idx, out_dir=OUT)

print("✅ Done. See ./analysis_paper/ for CSVs and Top-15 PDFs.")
