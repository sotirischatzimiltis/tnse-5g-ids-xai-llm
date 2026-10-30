import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

def feature_ablation(model, X, y, mode='single', metric='auc'):
    """
    Feature ablation analysis for LSTM model.
    
    Args:
        model (tf.keras.Model): Trained LSTM model
        X (np.ndarray): Input data with shape (n_samples, 3, 14)
        y (np.ndarray): Target values
        mode (str): Either 'single' (one feature at a time) or 'leave' (leave one feature out)
        metric (str): Metric to use ('auc' or 'accuracy')
        
    Returns:
        dict: Dictionary with feature ablation scores
    """
    # Get baseline performance with all features
    preds = model.predict(X)
    if metric == 'auc':
        baseline_score = roc_auc_score(y, preds)
    else:
        baseline_score = accuracy_score(y, (preds > 0.5).astype(int))
    
    n_timesteps, n_features = X.shape[1], X.shape[2]
    ablation_scores = {}
    
    # Ablate each feature at each timestep
    for t in range(n_timesteps):
        for f in range(n_features):
            feature_name = f"Timestep_{t}_Feature_{f}"
            X_ablated = X.copy()
            
            if mode == 'single':
                # Only include this feature (zero out others)
                mask = np.zeros_like(X)
                mask[:, t, f] = 1
                X_ablated = X * mask
            else:  # 'leave'
                # Leave this feature out (zero it out)
                X_ablated[:, t, f] = 0
            
            # Calculate score with ablated data
            preds_ablated = model.predict(X_ablated)
            if metric == 'auc':
                ablated_score = roc_auc_score(y, preds_ablated)
            else:
                ablated_score = accuracy_score(y, (preds_ablated > 0.5).astype(int))
            
            if mode == 'single':
                # Importance is the performance with only this feature
                ablation_scores[feature_name] = ablated_score
            else:  # 'leave'
                # Importance is the decrease in performance without this feature
                ablation_scores[feature_name] = baseline_score - ablated_score
    
    return ablation_scores

def visualize_ablation_scores(ablation_scores, mode, n_timesteps=3, n_features=14):
    """
    Visualize feature ablation scores for LSTM input.
    
    Args:
        ablation_scores (dict): Dictionary with ablation scores
        mode (str): Either 'single' or 'leave'
        n_timesteps (int): Number of timesteps
        n_features (int): Number of features
    """
    # Reshape scores for visualization
    ablation_matrix = np.zeros((n_timesteps, n_features))
    
    for feature_name, score in ablation_scores.items():
        t, f = map(int, feature_name.replace('Timestep_', '').replace('Feature_', '').split('_'))
        ablation_matrix[t, f] = score
    plt.figure(figsize=(12, 7))
    title = f"Feature Ablation ({'Only One Feature' if mode == 'single' else 'Leave One Out'})"
    label = 'Performance score' if mode == 'single' else 'Importance (decrease in performance)'
    
    im = plt.imshow(ablation_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label=label)
    
    plt.yticks(range(n_timesteps), [f'Timestep {i}' for i in range(n_timesteps)])
    plt.xticks(range(n_features), [f'Feature {i}' for i in range(n_features)], rotation=45, ha='right')
    
    plt.title(title)
    plt.tight_layout()
    
    # Top 10 most important features
    flat_ablation = [(k, v) for k, v in ablation_scores.items()]
    
    # Sort by importance
    if mode == 'single':
        flat_ablation.sort(key=lambda x: x[1], reverse=True)  # Higher is better
    else:
        flat_ablation.sort(key=lambda x: x[1], reverse=True)  # Higher drop is more important
    
    top_k = 10
    top_features = flat_ablation[:top_k]
    
    plt.figure(figsize=(10, 6))
    plt.bar([item[0] for item in top_features], [item[1] for item in top_features])
    plt.xticks(rotation=90)
    plt.title(f'Top {top_k} Features by Ablation Score')
    plt.ylabel(label)
    plt.tight_layout()
    plt.show()
    #return plt

# Example usage

model = load_model('../saved_models/lstm_model_GTPU.h5')  # load the model
X_flattened = pd.read_csv("cumulative_X_test.csv")  # load data
X = X_flattened.values.reshape(-1, 3, 14)
y = pd.read_csv("cumulative_y_test.csv")

ablation_scores = feature_ablation(model, X, y, mode='leave')
visualize_ablation_scores(ablation_scores, mode='leave')
print("after visualization")
