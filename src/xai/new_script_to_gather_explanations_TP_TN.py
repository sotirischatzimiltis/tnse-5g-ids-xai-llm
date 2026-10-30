#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 14 explainability instances:
- 10 for Zero-Shot (5 TN + 5 TP)
- 4 for Few-Shot (1 FP + 1 FN + 1 extra TN + 1 extra TP)

Artifacts saved:
• input_raw_*.csv           → raw input values per instance (bitrates in Mbps)
• shap_*.csv/.pdf           → SHAP heatmaps (scaled, class 1)
• lime_*.csv/.html/.pdf     → LIME (predicted class; rule thresholds converted to RAW with bitrates in Mbps; plot shows Top-10 only)
• shap_global_heatmap.csv/.pdf → global mean |SHAP| (scaled, class 1)
• feature_stats_raw.csv     → raw-space stats (bitrates in Mbps)
• prompts_manifest.csv
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from lime.lime_tabular import RecurrentTabularExplainer
import shap
import joblib

# -------------------------------
# Paths & constants
# -------------------------------
MODEL_PATH    = '../saved_models/day_5_GTPU_non_resampled_model.h5'
FEATURES_CSV  = '../features.csv'        # scaled, flattened [N, 42]
LABELS_CSV    = '../labels.csv'
SCALER_PATH   = '../robust_scaler.pkl'   # RobustScaler fit on 14 features (or 42)
OUT_DIR       = './prompt_assets'
SEED          = 42

# -------------------------------
# Feature setup
# -------------------------------
features = [
    'epre', 'pusch_snr', 'p_ue', 'ul_mcs', 'cqi', 'ul_bitrate',
    'dl_mcs', 'dl_retx', 'ul_tx', 'dl_tx', 'ul_retx', 'dl_bitrate',
    'dl_err', 'ul_err'
]
T = 3
F = len(features)
flattened_feature_names = [f"{feat}_T{t}" for t in range(T) for feat in features]

# Bitrate conversion for RAW outputs
BITRATE_FEATURES = {'ul_bitrate', 'dl_bitrate'}
MBPS_DIVISOR = 1e6

# -------------------------------
# Scaled → Raw threshold converter (no "(raw)" suffix); bitrates in Mbps
# -------------------------------
def make_scaled_to_raw_converter_no_tag(scaler, features, T):
    n_in = getattr(scaler, "n_features_in_", None)
    F = len(features)

    def step_index_from_tback(t_back: int) -> int:
        return (T - 1) - t_back  # t-2->0, t-1->1, t-0->2

    def scaled_to_raw_value(base_feat: str, t_back: int, z: float):
        feat_idx = features.index(base_feat)
        if n_in == F:
            median = float(scaler.center_[feat_idx])
            iqr    = float(scaler.scale_[feat_idx])
        elif n_in == T * F:
            ti = step_index_from_tback(t_back)
            col_index = ti * F + feat_idx
            median = float(scaler.center_[col_index])
            iqr    = float(scaler.scale_[col_index])
        else:
            raise ValueError("Scaler dimension mismatch.")
        raw = (z * iqr) + median
        # Convert bitrates to Mbps for rule text
        if base_feat in BITRATE_FEATURES:
            raw = raw / MBPS_DIVISOR
        return raw

    def normalize_ttokens(s: str) -> str:
        s = re.sub(r"\bat\s+t-", "_t-", s, flags=re.IGNORECASE)
        s = re.sub(r"\bt-", "_t-", s, flags=re.IGNORECASE)
        return s

    NUM  = r"([\-+]?\d*\.?\d+(?:[eE][\-+]?\d+)?)"
    OP   = r"([<>]=?)"
    FEAT = r"([A-Za-z0-9_]+)_t-(\d+)"

    pat_range_num_feat_num = re.compile(rf"^\s*{NUM}\s*{OP}\s*{FEAT}\s*{OP}\s*{NUM}\s*$")
    pat_single_feat_num    = re.compile(rf"^\s*{FEAT}\s*{OP}\s*{NUM}\s*$")
    pat_single_num_feat    = re.compile(rf"^\s*{NUM}\s*{OP}\s*{FEAT}\s*$")

    def convert(text: str) -> str:
        s = normalize_ttokens(text.strip())

        m = pat_range_num_feat_num.match(s)
        if m:
            a_str, op1, feat, tback_str, op2, b_str = m.groups()
            t_back = int(tback_str)
            a_raw = scaled_to_raw_value(feat, t_back, float(a_str))
            b_raw = scaled_to_raw_value(feat, t_back, float(b_str))
            return f"{a_raw:.6g} {op1} {feat}_t-{t_back} {op2} {b_raw:.6g}"

        m = pat_single_feat_num.match(s)
        if m:
            feat, tback_str, op, a_str = m.groups()
            t_back = int(tback_str)
            a_raw = scaled_to_raw_value(feat, t_back, float(a_str))
            return f"{feat}_t-{t_back} {op} {a_raw:.6g}"

        m = pat_single_num_feat.match(s)
        if m:
            a_str, op, feat, tback_str = m.groups()
            t_back = int(tback_str)
            a_raw = scaled_to_raw_value(feat, t_back, float(a_str))
            return f"{a_raw:.6g} {op} {feat}_t-{t_back}"

        return text

    return convert

# -------------------------------
# Setup & load data
# -------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH, compile=True)

X_flat = pd.read_csv(FEATURES_CSV)                 # [N, 42] scaled
y_true = pd.read_csv(LABELS_CSV).values.ravel().astype(int)
assert X_flat.shape[1] == T * F

# Reshape (scaled) for model/explainers
X_flat_scaled = X_flat.values
X_scaled = X_flat_scaled.reshape(-1, T, F)

# Inverse-transform to RAW for human-facing CSVs
if scaler.n_features_in_ == F:
    X_raw = np.empty_like(X_scaled)
    for t in range(T):
        X_raw[:, t, :] = scaler.inverse_transform(X_scaled[:, t, :])
    X_flat_raw = X_raw.reshape(-1, T * F)
elif scaler.n_features_in_ == T * F:
    X_flat_raw = scaler.inverse_transform(X_flat_scaled)
    X_raw = X_flat_raw.reshape(-1, T, F)
else:
    raise ValueError("Scaler shape mismatch with data.")

scaled_rule_to_raw = make_scaled_to_raw_converter_no_tag(scaler, features, T)

# Predictions (scaled)
y_pred_prob = model.predict(X_scaled, verbose=0).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# -------------------------------
# Choose instances
# -------------------------------
rng = np.random.default_rng(SEED)
tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

chosen = np.concatenate([
    rng.choice(tn_idx, size=min(5, len(tn_idx)), replace=False),
    rng.choice(tp_idx, size=min(5, len(tp_idx)), replace=False),
    rng.choice(fp_idx, size=1, replace=False) if len(fp_idx) else np.array([], dtype=int),
    rng.choice(fn_idx, size=1, replace=False) if len(fn_idx) else np.array([], dtype=int),
    rng.choice(np.setdiff1d(tn_idx, tn_idx[:5]), size=1, replace=False) if len(tn_idx) > 5 else rng.choice(tn_idx, size=1, replace=False),
    rng.choice(np.setdiff1d(tp_idx, tp_idx[:5]), size=1, replace=False) if len(tp_idx) > 5 else rng.choice(tp_idx, size=1, replace=False),
])

# -------------------------------
# Explainability setup
# -------------------------------
def predict_fn(input_2d):
    r = input_2d.reshape(-1, T, F)
    p1 = model.predict(r, verbose=0).ravel()[:, None]
    return np.hstack((1 - p1, p1))

lime_explainer = RecurrentTabularExplainer(
    training_data=X_scaled,                 # scaled 3D
    mode="classification",
    feature_names=features,
    class_names=["Normal", "Attack"],
    random_state=SEED
)

background = pd.DataFrame(X_flat_scaled).iloc[: min(100, len(X_flat_scaled))]
shap_explainer = shap.KernelExplainer(predict_fn, background)

def shap_to_class1_2d(shap_vals):
    # Return (T,F) for class 1
    if isinstance(shap_vals, (list, tuple)):
        arr = np.asarray(shap_vals[0])     # (n, 42, 2) for KernelExplainer
    else:
        arr = np.asarray(shap_vals)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        cls1 = arr[:, :, 1]
    else:
        raise ValueError("Unexpected SHAP output shape.")
    return cls1.reshape(T, F)


def plot_lime_style_bar_top10(feature_contrib_pairs, filename=None):
    # feature_contrib_pairs: list[(feature_str_raw, contribution)]
    # keep Top-10 by |contribution|
    df = pd.DataFrame(feature_contrib_pairs, columns=["Feature", "Contribution"])
    df = df.reindex(df["Contribution"].abs().sort_values(ascending=False).index).head(10)

    feats = [f.replace('t-', 'T') for f in df["Feature"]]
    vals  = df["Contribution"].values
    colors = ['tab:orange' if v > 0 else 'tab:blue' for v in vals]

    fig, ax = plt.subplots(figsize=(3.0, 3.5))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=0.24)

    for i, (f, v) in enumerate(zip(feats, vals)):
        ax.text(v + 0.005 if v > 0 else v - 0.005,
                i - 0.15, f"{abs(v):.2f}",
                ha='left' if v > 0 else 'right', va='top',
                color='tab:orange' if v > 0 else 'tab:blue', fontsize=12, fontweight='bold')
        ax.text(0, i-0.175, f, fontsize=10, color='black',
                ha='left' if v > 0 else 'right', va='bottom')

    ax.axvline(0, color='black')
    ax.set_yticks([]); ax.set_xticks([]); ax.invert_yaxis()
    ax.text(-0.1, -1, "Normal", fontsize=12, color='tab:blue', ha='right')
    ax.text( 0.1, -1, "Attack", fontsize=12, color='tab:orange', ha='left')
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    if filename: plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

# -------------------------------
# Save helpers
# -------------------------------
def _apply_bitrate_mbps_df(df: pd.DataFrame) -> pd.DataFrame:
    # Divide bitrate columns by 1e6 if present
    for col in df.columns:
        base = col.split('_T')[0] if '_T' in col else col
        if base in BITRATE_FEATURES:
            df[col] = df[col] / MBPS_DIVISOR
    return df

def save_input_raw_csv(idx, out_path):
    df = pd.DataFrame(X_raw[idx], columns=features, index=[f"T{t}" for t in range(T)])
    # convert bitrates to Mbps for raw output
    for b in BITRATE_FEATURES:
        if b in df.columns:
            df[b] = df[b] / MBPS_DIVISOR
    df.to_csv(out_path)

def save_shap_local(idx, out_csv, out_pdf):
    vals = shap_explainer.shap_values(X_flat_scaled[idx:idx+1])
    cls1 = shap_class_array(vals, cls=1)          # (1, 42)
    shap_mat = cls1.reshape(T, F)                 # (T, F)

    pd.DataFrame(shap_mat, columns=features, index=[f"T{t}" for t in range(T)]).to_csv(out_csv)

    plt.figure(figsize=(4, 2.5))
    plt.imshow(shap_mat, cmap='bwr', aspect='auto', vmin=-0.05, vmax=0.05)
    plt.xticks(range(F), features, rotation=45, ha='right', fontsize=14)
    plt.yticks(range(T), [f"T{t}" for t in range(T)], fontsize=10)
    #plt.title("Local SHAP (class 1, scaled)", fontsize=9)
    plt.tight_layout(); plt.savefig(out_pdf, bbox_inches='tight'); plt.close()


def save_lime_local(idx, out_csv, out_html, out_pdf):
    # EXACTLY like old script: predicted class; then convert rule strings to RAW (Mbps for bitrates)
    exp = lime_explainer.explain_instance(X_scaled[idx], predict_fn, num_features=42)
    contribs = exp.as_list()  # predicted class
    df = pd.DataFrame(contribs, columns=["Feature", "Contribution"])

    # Convert LIME rule text thresholds to RAW (bitrate thresholds shown in Mbps)
    df["Feature"] = df["Feature"].apply(scaled_rule_to_raw)

    # Save CSV & HTML (HTML will show LIME's native view, which stays scaled; that's normal)
    df.to_csv(out_csv, index=False)
    exp.save_to_file(out_html)

    # Plot Top-10 only (old compact style; rule text already in raw Mbps for bitrates)
    plot_lime_style_bar_top10(list(zip(df["Feature"], df["Contribution"])), filename=out_pdf)

def shap_class_array(vals, cls=1):
    """
    Returns a 2D array (n_samples, n_features) for the requested class.
    Works for both styles:
      - list of arrays: [array_for_class0, array_for_class1, ...]
      - single 3D array: (n_samples, n_features, n_classes)
    """
    if isinstance(vals, (list, tuple)):
        # Typical KernelExplainer multiclass output
        if len(vals) <= cls:  # fallback if only one array returned
            return np.asarray(vals[0])
        return np.asarray(vals[cls])
    else:
        arr = np.asarray(vals)
        if arr.ndim == 3 and arr.shape[-1] > cls:
            return arr[:, :, cls]
        elif arr.ndim == 2:
            return arr
        else:
            raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def save_global_shap(out_csv, out_pdf=None, out_png=None):
    """
    Global mean |SHAP| (class=1) heatmap styled like the example:
      - bwr colormap, vmin=0, vmax=max(mean|SHAP|)
      - colorbar labeled 'SHAP Value'
      - x labels rotated 45°, y labels: T0, T1, T2
    """
    n_bg = min(400, len(X_flat_scaled))
    vals = shap_explainer.shap_values(X_flat_scaled[:n_bg])
    cls1 = shap_class_array(vals, cls=1)                 # (n_bg, 42)
    mean_abs = np.abs(cls1).mean(axis=0).reshape(T, F)   # (3, 14)

    # Save CSV
    pd.DataFrame(mean_abs, columns=features, index=[f"T{t}" for t in range(T)]).to_csv(out_csv)

    # Styled heatmap
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    im = ax.imshow(mean_abs, cmap='bwr', aspect='auto', vmin=0.0, vmax=float(mean_abs.max()))
    ax.set_xticks(np.arange(F))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_yticks(np.arange(T))
    ax.set_yticklabels([f"T{t}" for t in range(T)])
    ax.tick_params(length=0)
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SHAP Value")
    plt.tight_layout()

    # Save
    if out_pdf:
        plt.savefig(out_pdf, bbox_inches='tight')
    if out_png:
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def save_feature_stats_raw(out_csv):
    # Flatten RAW then convert bitrate columns to Mbps before aggregations
    df_r = pd.DataFrame(X_raw.reshape(-1, T*F), columns=flattened_feature_names)
    for t in range(T):
        for b in BITRATE_FEATURES:
            col = f"{b}_T{t}"
            if col in df_r.columns:
                df_r[col] = df_r[col] / MBPS_DIVISOR

    # Class-wise mean/std on per-feature (averaged across timesteps)
    collapsed = {f: df_r[[f"{f}_T{t}" for t in range(T)]].mean(axis=1) for f in features}
    Z = pd.DataFrame(collapsed); Z["label"] = y_true
    rows = []
    for f in features:
        z0 = Z.loc[Z["label"] == 0, f]; z1 = Z.loc[Z["label"] == 1, f]
        rows.append([f, z0.mean(), z0.std(), z1.mean(), z1.std()])
    pd.DataFrame(rows, columns=["feature","normal_mean","normal_std","attack_mean","attack_std"]).to_csv(out_csv, index=False)



def _topk_flat_indices(values_2d, k):
    """Top-k flattened features by mean |SHAP|."""
    mean_abs = np.abs(values_2d).mean(axis=0)  # (n_features,)
    return np.argsort(mean_abs)[-k:][::-1]     # descending

def _topk_base_indices(values_2d, feature_names, k, T=3):
    """
    Pick top-k BASE features (e.g., 'ul_bitrate') by summing mean|SHAP| over T0..T2,
    then return all flattened indices belonging to those bases (ordered by contribution).
    """
    # split "feat_Tt" into base, step
    bases = [name.split("_T")[0] for name in feature_names]
    uniq = sorted(set(bases), key=bases.index)  # keep order of first appearance

    # mean |SHAP| per flattened feature
    mean_abs = np.abs(values_2d).mean(axis=0)
    # sum over the three steps for each base
    base_scores = {b: 0.0 for b in uniq}
    for j, b in enumerate(bases):
        base_scores[b] += mean_abs[j]

    # take top-k bases
    top_bases = [b for b, _ in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:k]]

    # collect all flattened indices that belong to those bases, sorted by their mean_abs
    chosen_idx = [j for j, b in enumerate(bases) if b in top_bases]
    chosen_idx = sorted(chosen_idx, key=lambda j: mean_abs[j], reverse=True)
    return np.array(chosen_idx), top_bases

def save_shap_beeswarm(out_pdf,
                       n_eval=800,
                       top_k=20,
                       mode="flat",     # "flat" or "base"
                       group_k=10,      # when mode="base": number of bases to choose
                       figsize=(3.25, 2.8),  # good for single-column width
                       dpi_png=None):
    """
    Make a publication-friendly beeswarm for class 1.
    mode="flat": show top_k flattened features (among 42).
    mode="base": pick top 'group_k' base features (14), and show all their timesteps.
    """
    # use a manageable subset (for speed/variance)
    n = min(n_eval, len(X_flat_scaled))
    vals = shap_explainer.shap_values(X_flat_scaled[:n])  # explainer already defined in your code
    cls1 = shap_class_array(vals, cls=1)                  # (n, 42)

    names = flattened_feature_names
    if mode == "flat":
        idx = _topk_flat_indices(cls1, top_k)
        vals_plot = cls1[:, idx]
        data_plot = X_flat_scaled[:n, idx]
        names_plot = [names[i] for i in idx]
    else:
        idx, top_bases = _topk_base_indices(cls1, names, group_k, T=T)
        vals_plot = cls1[:, idx]
        data_plot = X_flat_scaled[:n, idx]
        names_plot = [names[i] for i in idx]

    # Build SHAP Explanation for beeswarm
    expl = shap.Explanation(values=vals_plot,
                            data=data_plot,
                            feature_names=names_plot)

    # Matplotlib tweaks for papers (vector fonts, small text)
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7
    })

    fig = plt.figure(figsize=figsize)
    shap.plots.beeswarm(expl, max_display=len(names_plot), show=False)
    plt.tight_layout()
    # Save
    fig.savefig(out_pdf, bbox_inches="tight")
    if dpi_png:  # optional raster copy
        fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)
# ----------------------------------------------------------------------

# -------------------------------
# Main execution
# -------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# Converter (after scaler is known)
scaled_rule_to_raw = make_scaled_to_raw_converter_no_tag(scaler, features, T)

# Predictions already computed above
manifest_rows = []
for k, idx in enumerate(chosen, start=1):
    yy, pp, prob = int(y_true[idx]), int(y_pred[idx]), float(y_pred_prob[idx])
    tag = (["TN-ZS"]*5 + ["TP-ZS"]*5 + ["FP-FS","FN-FS","TN-extra-FS","TP-extra-FS"])[k-1]

    paths = {
        "input_raw": os.path.join(OUT_DIR, f"input_raw_{k}.csv"),
        "shap_csv":  os.path.join(OUT_DIR, f"shap_{k}.csv"),
        "shap_pdf":  os.path.join(OUT_DIR, f"shap_{k}.pdf"),
        "lime_csv":  os.path.join(OUT_DIR, f"lime_{k}.csv"),
        "lime_html": os.path.join(OUT_DIR, f"lime_{k}.html"),
        "lime_pdf":  os.path.join(OUT_DIR, f"lime_{k}.pdf"),
    }

    save_input_raw_csv(idx, paths["input_raw"])
    save_shap_local(idx, paths["shap_csv"], paths["shap_pdf"])
    save_lime_local(idx, paths["lime_csv"], paths["lime_html"], paths["lime_pdf"])

    manifest_rows.append({
        "prompt_id": k, "dataset_index": idx,
        "y_true": yy, "y_pred": pp, "proba_class1": prob, "error_type": tag, **paths
    })

# Global artifacts
save_global_shap(
    out_csv=os.path.join(OUT_DIR, "shap_global_heatmap.csv"),
    out_pdf=os.path.join(OUT_DIR, "shap_global_heatmap.pdf")
)
save_feature_stats_raw(os.path.join(OUT_DIR, "feature_stats_raw.csv"))

pd.DataFrame(manifest_rows).to_csv(os.path.join(OUT_DIR, "prompts_manifest.csv"), index=False)

save_shap_beeswarm(os.path.join(OUT_DIR, "shap_beeswarm_top10_flat.pdf"),
                   n_eval=800, top_k=10, mode="flat", figsize=(3.25, 2.8))

print("\n✅ All artifacts saved in:", OUT_DIR)
print("   - per-instance: input_raw_* (bitrates in Mbps), shap_*, lime_* (Top-10 plots; raw rule text with Mbps)")
print("   - global: shap_global_heatmap.csv/.pdf, feature_stats_raw.csv (bitrates in Mbps)")
print("   - manifest: prompts_manifest.csv")
