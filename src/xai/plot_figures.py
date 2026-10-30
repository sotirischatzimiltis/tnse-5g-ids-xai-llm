# This script reads the uploaded CSVs and generates several matplotlib figures
# (no custom colors/styles) so you can choose the one you prefer for the paper.
# Outputs are saved under /mnt/data/analysis_figures and provided as download links.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "analysis_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Helper to robustly read the collapsed CSVs whether the index column is named or not.
def read_collapsed_csv(path):
    df = pd.read_csv(path)
    # If there is a first unnamed column that looks like feature names, set it as index
    if df.columns[0].lower() in ["base_feature", "base_feat", "feature", "Unnamed: 0"] or not np.issubdtype(df[df.columns[0]].dtype, np.number):
        df = df.set_index(df.columns[0])
    # Try to detect column names
    cols = list(df.columns)
    # Standardize names
    rename_map = {}
    for c in cols:
        cl = c.lower()
        if "a_mean_abs" in cl and "fp_vs_tn" in cl:
            rename_map[c] = "A_mean_abs"
        elif "b_mean_abs" in cl and "fp_vs_tn" in cl:
            rename_map[c] = "B_mean_abs"
        elif "a_mean_abs" in cl and "fn_vs_tp" in cl:
            rename_map[c] = "A_mean_abs"
        elif "b_mean_abs" in cl and "fn_vs_tp" in cl:
            rename_map[c] = "B_mean_abs"
        elif "a_mean_abs" in cl and "delta" not in cl:
            rename_map[c] = "A_mean_abs"
        elif "b_mean_abs" in cl and "delta" not in cl:
            rename_map[c] = "B_mean_abs"
        elif "delta" in cl:
            rename_map[c] = "delta_A_minus_B"
    df = df.rename(columns=rename_map)
    # Ensure required columns exist
    if "delta_A_minus_B" not in df.columns:
        # If not present, compute if A/B are available
        if {"A_mean_abs","B_mean_abs"}.issubset(df.columns):
            df["delta_A_minus_B"] = df["A_mean_abs"] - df["B_mean_abs"]
        else:
            raise ValueError(f"Could not find expected columns in {path}: {df.columns.tolist()}")
    return df

fp_path = "analysis_paper/FP_vs_TN_collapsed.csv"
fn_path = "analysis_paper/FN_vs_TP_collapsed.csv"

fp = read_collapsed_csv(fp_path)
fn = read_collapsed_csv(fn_path)

# Sort by absolute delta for consistent top selections
fp_sorted = fp.reindex(fp["delta_A_minus_B"].abs().sort_values(ascending=False).index)
fn_sorted = fn.reindex(fn["delta_A_minus_B"].abs().sort_values(ascending=False).index)

# --- Figure 1/2: Horizontal bar of Top-15 |Δ| (simple, paper-ready) ---
def bar_top_abs_delta(df_sorted, title, fname, topN=15):
    top = df_sorted.head(topN)
    plt.figure(figsize=(8, 5))
    top["delta_A_minus_B"].iloc[::-1].plot(kind="barh")
    plt.xlabel("Δ mean |SHAP| (A − B), collapsed over time")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

f1 = bar_top_abs_delta(fp_sorted, "FP vs TN — Top |Δ mean |SHAP|| (collapsed)", "fp_bar_top15.pdf", topN=15)
f2 = bar_top_abs_delta(fn_sorted, "FN vs TP — Top |Δ mean |SHAP|| (collapsed)", "fn_bar_top15.pdf", topN=15)

# --- Figure 3/4: Grouped horizontal bars (A vs B) for Top-10 |Δ| ---
def grouped_horizontal_bars(df_sorted, title, fname, topN=10):
    top = df_sorted.head(topN).copy()
    # We'll plot A and B side-by-side for each feature
    idx = top.index[::-1]  # reverse for top at top
    y = np.arange(len(idx))
    width = 0.4
    plt.figure(figsize=(8, 5))
    plt.barh(y - width/2, top.loc[idx, "A_mean_abs"], height=width, label="Group A (errors)")
    plt.barh(y + width/2, top.loc[idx, "B_mean_abs"], height=width, label="Group B (correct)")
    plt.yticks(y, idx)
    plt.xlabel("Mean |SHAP| (collapsed over time)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

# Ensure A/B columns exist; if not, skip grouped bars
grouped_paths = []
if {"A_mean_abs","B_mean_abs"}.issubset(fp.columns):
    grouped_paths.append(grouped_horizontal_bars(fp_sorted, "FP vs TN — Group comparison (Top-10)", "fp_grouped_top10.pdf", 10))
if {"A_mean_abs","B_mean_abs"}.issubset(fn.columns):
    grouped_paths.append(grouped_horizontal_bars(fn_sorted, "FN vs TP — Group comparison (Top-10)", "fn_grouped_top10.pdf", 10))

# --- Figure 5/6: Lollipop (stem) plots of Top-20 Δ (signed) ---
def lollipop_signed_delta(df_sorted, title, fname, topN=20):
    top = df_sorted.head(topN).copy()
    x = np.arange(len(top.index))
    y = top["delta_A_minus_B"].values
    plt.figure(figsize=(9, 4.5))
    markerline, stemlines, baseline = plt.stem(x, y)
    # Do not set colors explicitly per tool guidance
    plt.xticks(x, top.index, rotation=45, ha="right")
    plt.ylabel("Δ mean |SHAP| (A − B)")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

f5 = lollipop_signed_delta(fp_sorted, "FP vs TN — Signed Δ (Top-20)", "fp_lollipop_top20.pdf", 20)
f6 = lollipop_signed_delta(fn_sorted, "FN vs TP — Signed Δ (Top-20)", "fn_lollipop_top20.pdf", 20)

# --- Export quick CSVs with the top selections (for reference) ---
fp_sorted.head(20).to_csv(os.path.join(OUT_DIR, "fp_top20.csv"))
fn_sorted.head(20).to_csv(os.path.join(OUT_DIR, "fn_top20.csv"))

def grouped_horizontal_bars_visible(df_sorted, title, fname, label_A, label_B):
    top = df_sorted.copy()
    idx = top.index[::-1]
    y = np.arange(len(idx))
    width = 0.4  # slightly larger bars

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.barh(y - width/2, top.loc[idx, "A_mean_abs"], height=width, label=label_A)
    ax.barh(y + width/2, top.loc[idx, "B_mean_abs"], height=width, label=label_B)

    # Larger fonts for readability in small figure
    ax.set_yticks(y)
    ax.set_yticklabels(idx, fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel("Mean |SHAP|", fontsize=9)
    #ax.set_title(title, fontsize=10)

    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

fp_bigger = grouped_horizontal_bars_visible(fp_sorted, "FP vs TN", "fp_grouped.pdf",
                                            "FPs", "TNs")
fn_bigger = grouped_horizontal_bars_visible(fn_sorted, "FN vs TP", "fn_grouped.pdf",
                                            "FNs", "TPs")







