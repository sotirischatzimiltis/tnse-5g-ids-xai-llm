from collections import OrderedDict
import numpy as np
import json
from tabulate import tabulate
import pandas as pd
from keras.models import load_model

def generate_prompt_json(stats_dict, lstm_input, prediction, lime_path, shap_path, shap_global_path, feature_names):
    # Convert LSTM input to list of dicts
    lstm_input_dict = [
        {"timestamp": f"T{t}", **{feature_names[i]: float(f"{val:.5f}") for i, val in enumerate(row)}}
        for t, row in enumerate(lstm_input)
    ]

    # Load CSV tables and clean column names
    def load_clean_csv(path):
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Timestamp"})
        return df

    lime_df = load_clean_csv(lime_path)
    shap_df = load_clean_csv(shap_path)
    shap_global_df = load_clean_csv(shap_global_path)

    prompt_json = OrderedDict({
        "general_statistics": stats_dict,
        "lstm_input_sequence": lstm_input_dict,
        "model_prediction": {
            "label": int(prediction),
            "meaning": "Anomalous" if prediction == 1 else "Normal"
        },
        "local_explanations": {
            "lime": lime_df.to_dict(orient="records"),
            "shap": shap_df.to_dict(orient="records")
        },
        "global_explanations": {
            "shap_global": shap_global_df.to_dict(orient="records")
        },
        "task": [
            "Provide a human-readable summary of the model's decision.",
            "Highlight which features or patterns most influenced the classification outcome.",
            "What is the likelihood that this instance is a misclassification?",
            "Finally, suggest any actionable mitigation steps for the network operator."
        ]
    })

    return prompt_json


# Example Usage:
if __name__ == "__main__":
    # Simulated feature stats
    feature_stats = \
        {
            'epre': {
                'normal': {'mean': 0.56951, 'std': 0.2032},
                'attack': {'mean': 0.47603, 'std': 0.15596}
            },
            'pusch_snr': {
                'normal': {'mean': 0.65606, 'std': 0.09969},
                'attack': {'mean': 0.62498, 'std': 0.07325}
            },
            'p_ue': {
                'normal': {'mean': 0.39212, 'std': 0.24644},
                'attack': {'mean': 0.37143, 'std': 0.16375}
            },
            'ul_mcs': {
                'normal': {'mean': 0.67751, 'std': 0.27754},
                'attack': {'mean': 0.58851, 'std': 0.27669}
            },
            'cqi': {
                'normal': {'mean': 0.80974, 'std': 0.2172},
                'attack': {'mean': 0.68552, 'std': 0.26487}
            },
            'ul_bitrate': {
                'normal': {'mean': 0.07661, 'std': 0.12149},
                'attack': {'mean': 0.09052, 'std': 0.09825}
            },
            'dl_mcs': {
                'normal': {'mean': 0.71582, 'std': 0.28027},
                'attack': {'mean': 0.58471, 'std': 0.34702}
            },
            'dl_retx': {
                'normal': {'mean': 0.04547, 'std': 0.07453},
                'attack': {'mean': 0.05405, 'std': 0.06362}
            },
            'ul_tx': {
                'normal': {'mean': 0.2864, 'std': 0.32424},
                'attack': {'mean': 0.36736, 'std': 0.18031}
            },
            'dl_tx': {
                'normal': {'mean': 0.16234, 'std': 0.21904},
                'attack': {'mean': 0.07902, 'std': 0.0434}
            },
            'ul_retx': {
                'normal': {'mean': 0.0864, 'std': 0.11402},
                'attack': {'mean': 0.16298, 'std': 0.13181}
            },
            'dl_bitrate': {
                'normal': {'mean': 0.01618, 'std': 0.02201},
                'attack': {'mean': 0.00064, 'std': 0.00049}
            },
            'dl_err': {
                'normal': {'mean': 0.00116, 'std': 0.00851},
                'attack': {'mean': 0.00745, 'std': 0.02624}
            },
            'ul_err': {
                'normal': {'mean': 0.00078, 'std': 0.00545},
                'attack': {'mean': 0.0233, 'std': 0.08653}
            }
        }

    # Simulated model output
    prediction = 1  # Anomalous

    # Define feature names and time steps
    feature_names = ['epre', 'pusch_snr', 'p_ue', 'ul_mcs', 'cqi', 'ul_bitrate',
                'dl_mcs', 'dl_retx', 'ul_tx', 'dl_tx', 'ul_retx', 'dl_bitrate',
                'dl_err', 'ul_err']
    # Generate and print prompt

    model = load_model('../saved_models/lstm_model_GTPU.h5')
    X_flattened = pd.read_csv("../XAI_python_scripts/cumulative_X_test.csv")
    y = pd.read_csv("../XAI_python_scripts/cumulative_y_test.csv")

    # Reshape to 3D input for LSTM
    X = X_flattened.values.reshape(-1, 3, 14)
    X_for_lime = X.reshape(X.shape[0], -1)
    y_true = y.values.flatten()

    # -------------------------------
    # 🔹 Prediction Function for Explainers
    # -------------------------------
    def predict_fn(input_2d):
        reshaped_input = input_2d.reshape(-1, 3, 14)
        proba_class_1 = model.predict(reshaped_input, verbose=0)
        proba_class_0 = 1 - proba_class_1
        return np.hstack((proba_class_0, proba_class_1))

    # Get predictions and predicted labels
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # -------------------------------
    # 🔹 Find TP, TN, FP, FN Indices
    # -------------------------------
    TP_idx = np.where((y_true == 1) & (y_pred == 1))[0][0]
    TN_idx = np.where((y_true == 0) & (y_pred == 0))[0][0]
    FP_idx = np.where((y_true == 0) & (y_pred == 1))[0][0]
    FN_idx = np.where((y_true == 1) & (y_pred == 0))[0][0]
    TP_idx_2 = np.where((y_true == 1) & (y_pred == 1))[0][100]
    TN_idx_2 = np.where((y_true == 0) & (y_pred == 0))[0][100]

    # 🔹 Generate prompts for TP, TN, FP, FN
    indices = [TN_idx, TP_idx, FN_idx, FP_idx, TN_idx_2, TP_idx_2]
    instance_ids = [1, 2, 3, 4, 5, 6]

    # Paths to explanation files
    lime_paths = [f"lime_{i}.csv" for i in range(1,7)]
    shap_paths = [f"shap_{i}.csv" for i in range(1,7)]
    shap_global_path = "shap_global_heatmap.csv"

    for i, idx, lime_path, shap_path in zip(instance_ids, indices, lime_paths, shap_paths):
        instance_input = X[idx]  # Shape: (3, 14)
        prediction = int(y_pred[idx])  # 0 or 1

        # Save as JSON
        prompt_json = generate_prompt_json(
            stats_dict=feature_stats,
            lstm_input=instance_input,
            prediction=prediction,
            lime_path=lime_path,
            shap_path=shap_path,
            shap_global_path=shap_global_path,
            feature_names=feature_names
        )

        with open(f"prompt_{i}.json", "w") as fjson:
            json.dump(prompt_json, fjson, indent=2)

        print(f"[✓] Saved prompt_{i}.json")

