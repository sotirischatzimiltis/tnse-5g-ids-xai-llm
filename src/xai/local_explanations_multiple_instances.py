import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from lime.lime_tabular import RecurrentTabularExplainer
import shap

# -------------------------------
# 🔹 Load Data and Model
# -------------------------------
model = load_model('../saved_models/day_5_GTPU.h5')
X_flattened = pd.read_csv("../features.csv")
y = pd.read_csv("../labels.csv")
y_mc = pd.read_csv("../multiclass_labels.csv")

# Define feature names and time steps
features = ['epre', 'pusch_snr', 'p_ue', 'ul_mcs', 'cqi', 'ul_bitrate',
            'dl_mcs', 'dl_retx', 'ul_tx', 'dl_tx', 'ul_retx', 'dl_bitrate',
            'dl_err', 'ul_err']
num_time_steps = 3
flattened_feature_names = [f"{feat}_T{t}" for t in range(num_time_steps) for feat in features]

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

# -------------------------------
# 🔹 Initialize LIME and SHAP
# -------------------------------
lime_explainer = RecurrentTabularExplainer(
    training_data=X,
    mode="classification",
    feature_names=features,
    class_names=["Normal", "Attack"],
    random_state=42
)

shap_explainer = shap.KernelExplainer(predict_fn, X_flattened[:100])

def save_shap_heatmap(instance_idx, label, filename):
    # Get SHAP values for one instance (class 1)
    shap_vals = shap_explainer.shap_values(X_flattened.iloc[instance_idx:instance_idx+1])
    shap_instance = shap_vals[0][:, 1]  # shape: (42,)
    #shap_instance_class_0 = shap_vals[0][:,1] # get the shap values for class 0
    # Reshape for heatmap: (3, 14)
    shap_matrix = shap_instance.reshape(3, 14)
    #shape_matrix_class_0 = shap_instance_class_0.reshape(3, 14)
    # Plot SHAP heatmap
    plt.figure(figsize=(4, 2.5))
    im = plt.imshow(shap_matrix, cmap='bwr', aspect='auto', vmin=-0.05, vmax=0.05)
    #plt.colorbar(label="SHAP Value", shrink=0.8)

    # Save as CSV
    shap_df = pd.DataFrame(shap_matrix, columns=features)
    shap_df.index = [f"T{i}" for i in range(3)]
    shap_df.to_csv(f"shap_{label.lower()}.csv")

    # Format axis
    plt.xticks(ticks=np.arange(14), labels=features, rotation=45, ha='right', fontsize=14)
    plt.yticks(ticks=np.arange(3), labels=['T0', 'T1', 'T2'], fontsize=12)
    plt.tick_params(axis='both', length=0)
    #plt.title(f'SHAP Heatmap – {label}', fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_shap_heatmap(TP_idx, "TP", "shap_tp.pdf")
save_shap_heatmap(TN_idx, "TN", "shap_tn.pdf")
save_shap_heatmap(FP_idx, "FP", "shap_fp.pdf")
save_shap_heatmap(FN_idx, "FN", "shap_fn.pdf")

def format_feature_name(feat_name):
    return feat_name.replace('t-', 'T')  # turns 't-2' into 'T2'

def plot_lime_style_bar(explanation, label, filename=None):
    feature_names, contributions = zip(*explanation.as_list())
    formatted_names = [f.replace('t-', 'T') for f in feature_names]
    values = list(contributions)
    colors = ['tab:orange' if val > 0 else 'tab:blue' for val in values]

    # Reduce height to pack bars closer
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # Smaller height
    y_pos = list(range(len(values)))

    # Thinner bars
    ax.barh(y_pos, values, color=colors, align='center', height=0.2)

    # Add only the value text (no feature name)
    for i, (feat, val) in enumerate(zip(formatted_names, values)):
        offset = 0.005
        ax.text(val + offset if val > 0 else val - offset,
                i - 0.15,
                f"{abs(val):.2f}",
                va='top',
                ha='left' if val > 0 else 'right',
                fontsize=12,
                color='tab:orange' if val > 0 else 'tab:blue',
                fontweight='bold')

        # Feature name depending on direction
        ax.text(0, i-0.175, feat,
                va='bottom',
                ha='left' if val > 0 else 'right',
                fontsize=12,
                color='black')

    # Styling
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()

    # Top labels
    ax.text(-0.1, -1, "Normal", fontsize=12, color='tab:blue', ha='right')
    ax.text(0.1, -1, "Attack", fontsize=12, color='tab:orange', ha='left')

    for spine in ax.spines.values():
        spine.set_visible(False)

    import matplotlib.patches as patches

    plt.tight_layout()
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()


def save_all_lime_plots():
    labels = ['TP', 'TN', 'FP', 'FN']
    indices = [TP_idx, TN_idx, FP_idx, FN_idx]

    for label, idx in zip(labels, indices):
        explanation = lime_explainer.explain_instance(
            X[idx],
            predict_fn,
            num_features=42 # or 42 for full
        )
        lime_data = pd.DataFrame(explanation.as_list(), columns=["Feature", "Contribution"])
        lime_data.to_csv(f"lime_{label.lower()}.csv", index=False)
        explanation.save_to_file(f"lime_{label.lower()}.html")
        plot_lime_style_bar(explanation, label, filename=f"lime_{label.lower()}.pdf")

save_all_lime_plots()
