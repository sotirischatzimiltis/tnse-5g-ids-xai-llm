# Explainable LSTM-based Anomaly Detection for DDoS in 5G, with LLM-generated Interpretations

Code accompanying our IEEE Transactions on Network Science and Engineering (TNSE)
paper. The pipeline detects DDoS attacks launched by malicious UEs in a 5G
network with an LSTM-based anomaly detector, explains its decisions with
post-hoc XAI methods, and uses Large Language Models to turn those
explanations into human-readable reports for network operators.

```
5G UE / cell metrics ──► time-series sequences ──► LSTM detector ──► XAI (SHAP, LIME, IG, ...) ──► LLM interpretation
```

## Repository structure

| Path | Contents |
|---|---|
| `data/` | Download instructions for the NCSRD-DS-5GDDoS v2.0 dataset (data not included) |
| `notebooks/01_preprocessing/` | Feature dropping, per-UE and per-attack dataset creation, input sequence generation |
| `notebooks/02_eda/` | Exploratory data analysis, KDE plots, feature statistics |
| `notebooks/03_training/` | LSTM training, TCN / Transformer / KNN / XGBoost baselines, imbalance handling, window-size and ratio fine-tuning, result analysis |
| `notebooks/04_xai/` | XAI experiments on the trained LSTM |
| `src/xai/` | Explainability scripts: DeepSHAP, TimeSHAP, LIME, Integrated Gradients, LRP, feature ablation, permutation importance, TP/TN/FP/FN explanation gathering, inference-time measurement, plotting |
| `notebooks/05_llm/` | Prompt creation, zero-/few-shot LLM API calls, FLOPs estimation, readability and BERTScore evaluation |
| `src/llm/` | Builders for the interpretation prompts and few-shot payloads |
| `prompts/` | Zero-shot and few-shot prompts and human-written reference interpretations |
| `models/` | Trained per-day LSTM models (`.h5`) with Platt-scaling calibrators and decision thresholds, XGBoost baselines, and the fitted `RobustScaler` |
| `figures/` | Figures used in the paper (FPR/FNR heatmaps, results, SHAP/LIME explanations) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset as described in [`data/README.md`](data/README.md), then run the
notebooks in numeric folder order. Notebooks were developed in a flat working
directory, so you may need to adjust the relative input/output paths at the top
of each notebook.

### LLM API keys

The notebooks in `notebooks/05_llm/` call OpenAI, DeepSeek, Mistral, Google
(Gemma) and OpenRouter models. Keys are read from environment variables. **Never
hardcode them.**

```bash
cp .env.example .env   # fill in your keys; .env is git-ignored
export $(grep -v '^#' .env | xargs)
```

## Citation

```bibtex
@article{chatzimiltis_tnse,
  title   = {TODO},
  author  = {Chatzimiltis, Sotiris and TODO},
  journal = {IEEE Transactions on Network Science and Engineering},
  year    = {TODO}
}
```
