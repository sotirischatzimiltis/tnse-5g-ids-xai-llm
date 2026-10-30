import pandas as pd
from tabulate import tabulate
import os
FEATURES = ['epre','pusch_snr','p_ue','ul_mcs','cqi','ul_bitrate',
            'dl_mcs','dl_retx','ul_tx','dl_tx','ul_retx','dl_bitrate','dl_err','ul_err']

def format_feature_stats_csv(path):
    df = pd.read_csv(path)
    df = df[["feature","normal_mean","normal_std","attack_mean","attack_std"]]
    df.iloc[:,1:] = df.iloc[:,1:].round(4)
    return tabulate(df.values.tolist(), headers=["Feature","Normal Mean","Normal Std","Attack Mean","Attack Std"], tablefmt="grid")

def format_matrix_csv(path):
    df = pd.read_csv(path, index_col=0)
    df = df[FEATURES].round(5)
    df.insert(0, "Timestep", df.index)
    return tabulate(df.values.tolist(), headers=["Timestep"]+FEATURES, tablefmt="grid")

def format_csv_table(path, title):
    try:
        df = pd.read_csv(path)
        df = df.round(5)
        return f"**{title}**\n" + tabulate(df.values.tolist(), headers=df.columns, tablefmt="grid")
    except Exception as e:
        return f"[Error reading {path}: {e}]"

def make_prompt(stats_tbl, input_tbl, pred, lime_tbl, shap_tbl, shap_global_tbl):
    label = "Anomalous" if pred==1 else "Normal"
    return f"""
You are a cybersecurity expert analyzing LSTM outputs for DDoS detection. Respond with 3 sections:
Anomaly Summary (3-6 bullet points), Misclassification Likelihood (1-2 bullets), and Mitigation Steps (2-4 bullets).

---

**Normalized General Data Distribution (Feature Statistics)**
{stats_tbl}

---

**Input Sequence to LSTM Model** (3 timesteps × 14 features)
{input_tbl}

---

**Model Prediction Output:** `{pred}` (*{label}*)

---

{lime_tbl}

{shap_tbl}

---

{shap_global_tbl}

---

**Task:**
Provide a human-readable summary of the model's decision.
Highlight which features or patterns most influenced the classification outcome.
What is the likelihood that this instance is a misclassification?
Finally, suggest actionable mitigation steps for the network operator.
""".strip()

# ---------- build 10 prompts ----------
man = pd.read_csv("../XAI_python_scripts/prompt_assets/prompts_manifest.csv")
stats_tbl = format_feature_stats_csv("../XAI_python_scripts/prompt_assets/feature_stats.csv")
shap_global_tbl = format_csv_table("../XAI_python_scripts/prompt_assets/shap_global_heatmap.csv", "Global SHAP Feature Importance")

# Ensure output directory
os.makedirs("Zero_Shot_Prompts_new", exist_ok=True)

for _, row in man.sort_values("prompt_id").iterrows():
    pid = int(row["prompt_id"])
    pred = int(row["y_pred"])
    input_tbl = format_matrix_csv(f"../XAI_python_scripts/prompt_assets/input_{pid}.csv")
    lime_tbl  = format_csv_table(f"../XAI_python_scripts/prompt_assets/lime_{pid}.csv",  "LIME Local Explanation")
    shap_tbl  = format_csv_table(f"../XAI_python_scripts/prompt_assets/shap_{pid}.csv",  "SHAP Local Heatmap")

    prompt_txt = make_prompt(stats_tbl, input_tbl, pred, lime_tbl, shap_tbl, shap_global_tbl)
    with open(f"Zero_Shot_Prompts_new/prompt_zero_shot_{pid}.txt","w", encoding="utf-8") as f:
        f.write(prompt_txt)

print("✅ Wrote 10 no-shot prompts: prompt_zero_shot_1..10.txt")
