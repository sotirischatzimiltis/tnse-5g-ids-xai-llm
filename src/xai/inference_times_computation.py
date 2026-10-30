import numpy as np
import pandas as pd
import time
from keras.models import load_model
from lime.lime_tabular import RecurrentTabularExplainer
import shap

# -------------------------------
# 🔹 Load Model and Data
# -------------------------------
model = load_model('../saved_models/lstm_model_GTPU.h5')
X_flattened = pd.read_csv("cumulative_X_test.csv")
y = pd.read_csv("cumulative_y_test.csv")

predict_call_counter = 0
flops_per_inference = 35500
# -------------------------------
# 🔹 Prepare Data
# -------------------------------
features = ['epre', 'pusch_snr', 'p_ue', 'ul_mcs', 'cqi', 'ul_bitrate',
            'dl_mcs', 'dl_retx', 'ul_tx', 'dl_tx', 'ul_retx', 'dl_bitrate',
            'dl_err', 'ul_err']
X = X_flattened.values.reshape(-1, 3, 14)
X_for_lime = X.reshape(X.shape[0], -1)
y_true = y.values.flatten()

# -------------------------------
# 🔹 Measure Model Inference Time
# -------------------------------
start_time = time.time()
y_pred_probs = model.predict(X, verbose=0)
inference_time_ms = (time.time() - start_time) * 1000
print(f"✅ Total inference time: {inference_time_ms:.2f} ms")
print(f"✅ Average inference time per instance: {inference_time_ms / X.shape[0]:.2f} ms")

# -------------------------------
# 🔹 Prediction Function
# -------------------------------
def predict_fn(input_2d):
    global predict_call_counter
    reshaped_input = input_2d.reshape(-1, 3, 14)
    batch_size = reshaped_input.shape[0]
    predict_call_counter += batch_size
    proba_class_1 = model.predict(reshaped_input, verbose=0)
    proba_class_0 = 1 - proba_class_1
    return np.hstack((proba_class_0, proba_class_1))

# -------------------------------
# 🔹 Identify an Example Index (TP)
# -------------------------------
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
TP_idx = np.where((y_true == 1) & (y_pred == 1))[0][0]

# -------------------------------
# 🔹 SHAP Inference Time
# -------------------------------
start_time = time.time()
shap_explainer = shap.KernelExplainer(predict_fn, X_flattened[:100])
_ = shap_explainer.shap_values(X_flattened.iloc[TP_idx:TP_idx+1])
shap_time_ms = (time.time() - start_time) * 1000
print(f"🟣 SHAP explanation time (1 instance): {shap_time_ms:.2f} ms")

print(f"📊 SHAP model calls: {predict_call_counter}")
print(f"📈 Estimated SHAP FLOPs: {predict_call_counter * flops_per_inference:,}")
predict_call_counter = 0

# -------------------------------
# 🔹 LIME Inference Time
# -------------------------------
lime_explainer = RecurrentTabularExplainer(
    training_data=X,
    mode="classification",
    feature_names=features,
    class_names=["Normal", "Attack"],
    random_state=42
)

start_time = time.time()
_ = lime_explainer.explain_instance(
    X[TP_idx],
    predict_fn,
    num_features=42
    # num_samples=1000
)
lime_time_ms = (time.time() - start_time) * 1000
print(f"🟠 LIME explanation time (1 instance): {lime_time_ms:.2f} ms")

print(f"📊 LIME model calls: {predict_call_counter}")
print(f"📈 Estimated LIME FLOPs: {predict_call_counter * flops_per_inference:,}")
