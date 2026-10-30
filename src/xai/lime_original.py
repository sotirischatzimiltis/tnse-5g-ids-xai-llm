import numpy as np
import pandas as pd
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
# Load model and data
model = load_model('../saved_models/lstm_model_GTPU.h5')
X_flattened = pd.read_csv("cumulative_X_test.csv")
y = pd.read_csv("cumulative_y_test.csv")

# Reshape into 3D for LSTM predictions
X = X_flattened.values.reshape(-1, 3, 14)

# Make sure the input shape is correct
print(f"X shape for model: {X.shape}")  # (samples, time_steps=3, features=14)

# ----- LIME expects 2D input, so we need to flatten each sample back -----
# This will be the shape (samples, 3*14 = 42)
X_for_lime = X.reshape(X.shape[0], -1)

# Define a prediction function that takes 2D input, reshapes it, and feeds to the model
def predict_fn(input_2d):
    reshaped_input = input_2d.reshape(-1, 3, 14)
    proba_class_1 = model.predict(reshaped_input)
    # Make it [p(class 0), p(class 1)] for each sample
    proba_class_0 = 1 - proba_class_1
    return np.hstack((proba_class_0, proba_class_1))


# Initialize LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=X_for_lime,
    mode="classification",
    feature_names=[f"t{t}_f{f}" for t in range(3) for f in range(14)],
    class_names=["Class 0", "Class 1"],
    discretize_continuous=True
)

# Pick a sample to explain (e.g. first one)
idx = 0
explanation = explainer.explain_instance(
    X_for_lime[idx],
    predict_fn,
    num_features=42
)

# Convert explanation to a dictionary
explanation_dict = dict(explanation.as_list())
# Or save as HTML
explanation.save_to_file("lime_explanation_sample.html")

