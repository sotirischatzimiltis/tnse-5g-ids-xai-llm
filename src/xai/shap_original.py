import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from keras.models import load_model
import seaborn as sns

# Load model and test data
# model = load_model('../saved_models/lstm_model_GTPU.h5')
# X_flattened = pd.read_csv("cumulative_X_test.csv")
# y = pd.read_csv("cumulative_y_test.csv")
#
# # Reshape for model: (samples, time_steps=3, features=14)
# X = X_flattened.values.reshape(-1, 3, 14)
#
# # Flatten for SHAP: (samples, 3*14 = 42)
# X_flat = X.reshape(X.shape[0], -1)
#
# # Define SHAP-compatible prediction function
# def predict_fn(input_2d):
#     reshaped = input_2d.reshape(-1, 3, 14)
#     proba_class_1 = model.predict(reshaped)
#     proba_class_0 = 1 - proba_class_1
#     return np.hstack((proba_class_0, proba_class_1))
#
# # Initialize SHAP KernelExplainer (use a subset for background)
# explainer = shap.KernelExplainer(predict_fn, X_flat[:100])  # small background for speed
#
# # Explain a single instance (you can loop this later)
# instance_idx = 0
# shap_values = explainer.shap_values(X_flat[instance_idx:instance_idx+1])
# print("SHAP output shape:", np.array(shap_values).shape)
# # SHAP values for the positive class (class 1)
# #shap_vals_instance = shap_values[1][0]  # shape: (42,)
# # Extract SHAP values for class 1
# shap_vals_instance = shap_values[0][:, 1]  # shape: (42,)
#
# # Reshape back to (3, 14) for heatmap
# shap_matrix = shap_vals_instance.reshape(3, 14)  # 3 time steps × 14 features
#
# # Optional: Feature names for axes
# feature_names = [f"f{i}" for i in range(14)]
# time_steps = [f"t{i}" for i in range(3)]
#
#
# plt.figure(figsize=(12, 4))
# plt.imshow(shap_matrix, cmap='bwr', aspect='auto')
#
# plt.colorbar(label="SHAP value")
# plt.title("SHAP Contributions Over Time and Features")
# plt.xlabel("Features")
# plt.ylabel("Time Steps")
# plt.xticks(ticks=np.arange(14), labels=feature_names)
# plt.yticks(ticks=np.arange(3), labels=time_steps)
# plt.grid(False)
# plt.tight_layout()
# plt.show()


# # Define feature names
features = ['epre', 'pusch_snr', 'p_ue', 'ul_mcs', 'cqi', 'ul_bitrate',
             'dl_mcs', 'dl_retx', 'ul_tx', 'dl_tx', 'ul_retx', 'dl_bitrate',
             'dl_err', 'ul_err']
#
# # Define time steps
num_time_steps = 3
#
# # Generate flattened feature names
flattened_feature_names = [f"{feat}_T{t}" for t in range(num_time_steps) for feat in features]
#
#
# normal_idx = np.where(y == 0)[0]
# attack_idx = np.where(y == 1)[0]
#
# # Sample up to 5000 from each (or less if your data is small)
# num_samples = min(len(normal_idx), len(attack_idx), 500)
#
# # Random but reproducible
# np.random.seed(42)
# normal_sample = np.random.choice(normal_idx, num_samples, replace=False)
# attack_sample = np.random.choice(attack_idx, num_samples, replace=False)
#
# # Combine and sort to keep it neat
# final_idx = np.concatenate([normal_sample, attack_sample])
# final_idx.sort()
#
# # Subset the data
# X_balanced = X_flat[final_idx]
# y_np = y.values.ravel()  # If y is a DataFrame with one column
# y_balanced = y_np[final_idx]
#
#
#
# # Compute SHAP values
# shap_values = explainer.shap_values(X_balanced)
# shap_vals_class1 = shap_values[:, :, 1]  # For class 1 (Attack)
#
# # Save SHAP values
# np.save("shap_values_class1.npy", shap_vals_class1)
#
# # (Optional) Save the input data too
# np.save("X_balanced.npy", X_balanced)
#
# # Wrap in SHAP Explanation object
# explanation = shap.Explanation(
#     values=shap_vals_class1,
#     data=X_balanced,
#     feature_names=flattened_feature_names
# )
#
# # Plot and save
# plt.figure()
# shap.plots.beeswarm(explanation, show=False)
# plt.tight_layout()
# plt.savefig("shap_global_beeswarm.pdf", format="pdf", bbox_inches='tight')
# plt.savefig("shap_global_beeswarm.png", format="png", dpi=300, bbox_inches='tight')
# plt.close()
#
# # Load saved data
# shap_vals_class1 = np.load("shap_values_class1.npy")
# X_flat_loaded = np.load("X_balanced.npy")
#
# # Rebuild SHAP Explanation object
# explanation = shap.Explanation(
#     values=shap_vals_class1,
#     data=X_flat_loaded,
#     feature_names=flattened_feature_names
# )
#
# # Replot
# # ✅ Plot only top 10 features
# plt.figure()
# shap.plots.beeswarm(explanation, max_display=10, show=False)
# plt.tight_layout()
# plt.savefig("shap_top10_beeswarm.pdf", format="pdf", bbox_inches='tight')
# plt.savefig("shap_top10_beeswarm.png", format="png", dpi=300, bbox_inches='tight')
# plt.close()



# # Load
shap_vals_class1 = np.load("shap_values_class1.npy")
X_balanced = np.load("X_balanced.npy")

# Mean absolute SHAP across all samples
mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
top10_idx = np.argsort(mean_abs_shap)[-10:][::-1]  # Top 10 indices

# Subset everything
shap_top10 = shap_vals_class1[:, top10_idx]
X_top10 = X_balanced[:, top10_idx]
top10_names = [flattened_feature_names[i] for i in top10_idx]
print(top10_names)
# Create new Explanation object
top10_expl = shap.Explanation(
    values=shap_top10,
    data=X_top10,
    feature_names=top10_names
)

# Plot without grouping anything
plt.figure()
shap.plots.beeswarm(top10_expl, show=False)
plt.tight_layout()
plt.savefig("shap_top10_exact.pdf", bbox_inches="tight")
plt.close()


## global heatmap:
num_time_steps = 3
num_features_per_step = 14

# Reshape SHAP values to (num_instances, time_steps, features_per_step)
shap_3d = shap_vals_class1.reshape(-1, num_time_steps, num_features_per_step)

# Compute mean absolute SHAP value per feature/time step
mean_abs_shap = np.mean(np.abs(shap_3d), axis=0)  # shape: (3, 14)

# Heatmap
# Plot SHAP heatmap
plt.figure(figsize=(6, 3.5))
im = plt.imshow(mean_abs_shap, cmap='bwr', aspect='auto', vmin=0.0, vmax=0.1)
plt.colorbar(label="SHAP Value", shrink=0.8)

# Format axis
plt.xticks(ticks=np.arange(14), labels=features, rotation=60, ha='right', fontsize=14)
plt.yticks(ticks=np.arange(3), labels=['T0', 'T1', 'T2'], fontsize=12)
plt.tick_params(axis='both', length=0)
#plt.title(f'SHAP Heatmap – {label}', fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.savefig("shap_global_heatmap.pdf", dpi=300, bbox_inches='tight')
plt.close()

# Save the global SHAP heatmap values to CSV
shap_df = pd.DataFrame(mean_abs_shap, columns=features)
shap_df.index = [f"T{i}" for i in range(num_time_steps)]
shap_df.to_csv("shap_global_heatmap.csv")
print("[✓] Saved global SHAP heatmap to shap_global_heatmap.csv")
