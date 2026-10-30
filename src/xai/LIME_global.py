import numpy as np
import pandas as pd
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from collections import defaultdict

# Load model and data
model = load_model('../saved_models/lstm_model_GTPU.h5')
X_flattened = pd.read_csv("cumulative_X_test.csv")
y = pd.read_csv("cumulative_y_test.csv")

# Reshape into 3D for LSTM predictions
X = X_flattened.values.reshape(-1, 3, 14)

# Flatten for LIME
X_for_lime = X.reshape(X.shape[0], -1)


# Define prediction function
def predict_fn(input_2d):
    reshaped_input = input_2d.reshape(-1, 3, 14)
    proba_class_1 = model.predict(reshaped_input)
    proba_class_0 = 1 - proba_class_1
    return np.hstack((proba_class_0, proba_class_1))


# Initialize LimeTabularExplainer
feature_names = [f"t{t}_f{f}" for t in range(3) for f in range(14)]
explainer = LimeTabularExplainer(
    training_data=X_for_lime,
    mode="classification",
    feature_names=feature_names,
    class_names=["Class 0", "Class 1"],
    discretize_continuous=True
)


# Global explanation aggregation
def get_global_explanation(X_for_lime, predict_fn, num_samples=50, num_features=10):
    # Aggregate feature importances
    global_importance = defaultdict(lambda: {"pos": 0, "neg": 0, "count": 0})

    # Randomly sample instances
    sample_indices = np.random.choice(len(X_for_lime), min(num_samples, len(X_for_lime)), replace=False)

    for idx in sample_indices:
        explanation = explainer.explain_instance(
            X_for_lime[idx],
            predict_fn,
            num_features=num_features
        )

        # Aggregate feature importances
        for feature, importance in explanation.as_list():
            global_importance[feature]["count"] += 1
            if importance > 0:
                global_importance[feature]["pos"] += importance
            else:
                global_importance[feature]["neg"] += abs(importance)

    # Calculate average importances
    global_summary = []
    for feature, data in global_importance.items():
        avg_pos = data["pos"] / data["count"] if data["count"] > 0 else 0
        avg_neg = data["neg"] / data["count"] if data["count"] > 0 else 0
        global_summary.append({
            "feature": feature,
            "positive_importance": avg_pos,
            "negative_importance": avg_neg,
            "occurrence": data["count"] / num_samples
        })

    # Sort by overall importance
    global_summary.sort(key=lambda x: abs(x["positive_importance"]) + abs(x["negative_importance"]), reverse=True)

    return global_summary


# Generate global explanation
global_explanation = get_global_explanation(X_for_lime, predict_fn)

# Print global explanation
print("Global Feature Importance:")
for item in global_explanation[:10]:  # Top 10 features
    print(f"Feature: {item['feature']}")
    print(f"  Positive Importance: {item['positive_importance']:.4f}")
    print(f"  Negative Importance: {item['negative_importance']:.4f}")
    print(f"  Occurrence Rate: {item['occurrence']:.2%}")
    print()

# Optional: Visualize global feature importance
plt.figure(figsize=(10, 6))
features = [item['feature'] for item in global_explanation[:10]]
pos_importances = [item['positive_importance'] for item in global_explanation[:10]]
neg_importances = [-item['negative_importance'] for item in global_explanation[:10]]

plt.barh(features, pos_importances, label='Positive Importance', color='green')
plt.barh(features, neg_importances, label='Negative Importance', color='red')
plt.xlabel('Feature Importance')
plt.title('Global Feature Importance from LIME')
plt.legend()
plt.tight_layout()
plt.show()