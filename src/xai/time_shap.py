import numpy as np
import pandas as pd
import timeshap
import shap
import matplotlib.pyplot as plt
from keras.models import load_model

# Load model and test data
model = load_model('../saved_models/lstm_model_GTPU.h5')
X_flattened = pd.read_csv("cumulative_X_test.csv")
y = pd.read_csv("cumulative_y_test.csv")

# Reshape for model: (samples, time_steps=3, features=14)
X = X_flattened.values.reshape(-1, 3, 14)

# Flatten for SHAP: (samples, 3*14 = 42)
X_flat = X.reshape(X.shape[0], -1)


# Define TimeShap-compatible prediction function
def predict_fn(input_2d):
    reshaped = input_2d.reshape(-1, 3, 14)
    proba_class_1 = model.predict(reshaped)
    proba_class_0 = 1 - proba_class_1
    return np.hstack((proba_class_0, proba_class_1))


# Global TimeShap Explanation
def global_timeshap_explanation(X, model, predict_fn):
    # Use TimeShap for global explanation
    global_explainer = timeshap.KernelExplainer(predict_fn, X[:100])

    # Compute global SHAP values for first 100 samples
    global_shap_values = global_explainer.shap_values(X[:100])

    # Plot global importance
    plt.figure(figsize=(15, 6))

    # Beeswarm plot for global feature importance
    shap.plots.beeswarm(
        shap.Explanation(
            values=global_shap_values[0][:, 1],  # SHAP values for positive class
            data=X[:100],
            feature_names=[f"t{t}_f{f}" for t in range(3) for f in range(14)]
        )
    )

    # Heatmap of average absolute SHAP values
    avg_abs_shap = np.abs(global_shap_values[0][:, 1]).mean(axis=0)
    avg_shap_matrix = avg_abs_shap.reshape(3, 14)

    plt.figure(figsize=(12, 4))
    plt.imshow(avg_shap_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label="Average Absolute SHAP Value")
    plt.title("Global Feature Importance Across Time Steps")
    plt.xlabel("Features")
    plt.ylabel("Time Steps")
    plt.xticks(ticks=np.arange(14), labels=[f"f{i}" for i in range(14)])
    plt.yticks(ticks=np.arange(3), labels=[f"t{i}" for i in range(3)])
    plt.tight_layout()
    plt.show()

    return global_shap_values


# Perform global explanation
global_shap_values = global_timeshap_explanation(X_flat, model, predict_fn)

# Optional: Print global feature importance
feature_names = [f"t{t}_f{f}" for t in range(3) for f in range(14)]
global_importance = np.mean(np.abs(global_shap_values[0][:, 1]), axis=0)
sorted_indices = np.argsort(global_importance)[::-1]
print("\nGlobal Feature Importance:")
for idx in sorted_indices:
    print(f"{feature_names[idx]}: {global_importance[idx]}")