import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import shap
import tensorflow as tf
import lime
# Read Data and Load Model
model = load_model('../saved_models/lstm_model_GTPU.h5')
X_flattened = pd.read_csv("cumulative_X_test.csv")
y = pd.read_csv("cumulative_y_test.csv")

# Define feature names
features = ['epre', 'pusch_snr', 'p_ue', 'ul_mcs', 'cqi', 'ul_bitrate',
            'dl_mcs', 'dl_retx', 'ul_tx', 'dl_tx', 'ul_retx', 'dl_bitrate',
            'dl_err', 'ul_err']

# Define time steps
num_time_steps = 3

# Generate flattened feature names
flattened_feature_names = [f"{feat}_T{t}" for t in range(num_time_steps) for feat in features]

# ✅ Check the result
print(flattened_feature_names)

# Reshape into 3D for LSTM predictions
X = X_flattened.values.reshape(-1, 3, 14)
X_for_lime = X.reshape(X.shape[0], -1)


# Define a prediction function that takes 2D input, reshapes it,
# and feeds to the model
def predict_fn(input_2d):
    reshaped_input = input_2d.reshape(-1, 3, 14)
    proba_class_1 = model.predict(reshaped_input)
    proba_class_0 = 1 - proba_class_1
    return np.hstack((proba_class_0, proba_class_1))


# LIME LOCAL EXPLAINER

# Create detailed feature names for LIME
detailed_feature_names = []
for t in range(3):
    detailed_feature_names.extend([f"T{t}_{feature}" for feature in features])

# Initialize LimeTabularExplainer
# explainer = LimeTabularExplainer(
#     training_data=X_for_lime,
#     mode="classification",
#     feature_names=detailed_feature_names,
#     class_names=["Class 0", "Class 1"],
#     discretize_continuous=True
# )
explainer = lime.lime_tabular.RecurrentTabularExplainer(
    training_data=X,
    mode="classification",
    feature_names=features,
    class_names=["Normal", "Attack"],
)

idx = 0
explanation = explainer.explain_instance(
    X[idx],
    predict_fn,
    num_features=42
)

# Pick a sample to explain (e.g. first one)
# idx = 0
# explanation = explainer.explain_instance(
#     X_for_lime[idx],
#     predict_fn,
#     num_features=42
# )

# Convert explanation to a dictionary
explanation_dict = dict(explanation.as_list())

# Print the explanation with clear feature names
print("LIME Explanation for the selected instance:")
for feature, contribution in explanation_dict.items():
    print(f"{feature}: {contribution}")

# Save as HTML
explanation.save_to_file("lime_explanation_sample.html")

# SHAP

# -------------------------------
# 🧠 Step 1: Initialize SHAP Explainer
# -------------------------------
background_data = X_flattened[:100]  # Use subset of training data as background
shap_explainer = shap.KernelExplainer(predict_fn, background_data)

# -------------------------------
# 🔍 Step 2: Explain a Single Instance
# -------------------------------
instance_idx = 1
shap_values = shap_explainer.shap_values(X_flattened[instance_idx:instance_idx + 1])

# Extract SHAP values for class 1 (attack)
shap_vals_instance = shap_values[0][:, 1]  # Shape: (42,)
expected_value = shap_explainer.expected_value[1]

print("SHAP Values Shape:", shap_vals_instance.shape)
print("Expected Value (class 1):", expected_value)

# -------------------------------
# 📊 Step 3: Force Plot (Static)
# -------------------------------

# shap.force_plot(
#     expected_value,
#     shap_vals_instance,
#     feature_names=flattened_feature_names,
#     matplotlib=True
# )

# Create force plot (interactive, HTML format)
force_plot = shap.force_plot(
    expected_value,
    shap_vals_instance,
    feature_names=flattened_feature_names
)

# Save as HTML file
shap.save_html("force_plot.html", force_plot)


# -------------------------------
# 🌊 Step 4: Waterfall Plot
# -------------------------------
explanation = shap.Explanation(
    values=shap_vals_instance,
    base_values=expected_value,
    data=X_flattened.iloc[instance_idx].values,
    feature_names=flattened_feature_names
)
# shap.plots.waterfall(explanation)
plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.title("SHAP Waterfall Plot")
plt.tight_layout()
plt.savefig("waterfall_plot.png", dpi=300)
plt.close()
# -------------------------------
# 🔀 Step 5: Decision Plot
# -------------------------------
plt.figure()
shap.decision_plot(
    base_value=expected_value,
    shap_values=shap_vals_instance,
    features=X_flattened.iloc[instance_idx].values,
    feature_names=flattened_feature_names,
    show=False
)
plt.title("SHAP Decision Plot")
plt.tight_layout()
plt.savefig("decision_plot.png", dpi=300)
plt.close()
# -------------------------------
# 🔥 Step 6: Heatmap Visualization (3D Time-Series)
# -------------------------------
# Reshape SHAP values: (42,) → (3, 14) for time-step vs feature view
shap_matrix = shap_vals_instance.reshape(3, 14)

# Time-step labels for y-axis
time_steps = ['Timestep 0', 'Timestep 1', 'Timestep 2']

# Plot SHAP matrix as heatmap
plt.figure(figsize=(15, 6))
plt.imshow(shap_matrix, cmap='bwr', aspect='auto')
plt.colorbar(label="SHAP value")
plt.title("SHAP Contributions Over Time and Features")

# Feature labels for x-axis
plt.xticks(ticks=np.arange(14), labels=features, rotation=45, ha='right')
plt.yticks(ticks=np.arange(3), labels=time_steps)

plt.grid(False)
plt.tight_layout()
#plt.show()
plt.savefig("shap_heatmap.png", dpi=300)
plt.close()
