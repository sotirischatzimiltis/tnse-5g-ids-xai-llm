import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import load_model
import pandas as pd

def permutation_feature_importance(model, X, y, metric='auc', n_repeats=5):
    """
    Calculate permutation feature importance for LSTM model.
    
    Args:
        model (tf.keras.Model): Trained LSTM model
        X (np.ndarray): Input data with shape (n_samples, 3, 14)
        y (np.ndarray): Target values
        metric (str): Metric to use ('auc' or 'accuracy')
        n_repeats (int): Number of times to repeat the permutation
        
    Returns:
        dict: Dictionary with feature importance scores for each feature
    """
    # Get baseline performance
    preds = model.predict(X)
    if metric == 'auc':
        baseline_score = roc_auc_score(y, preds)
    else:
        baseline_score = accuracy_score(y, (preds > 0.5).astype(int))
    
    n_timesteps, n_features = X.shape[1], X.shape[2]
    importance_scores = {}
    
    # For each feature at each timestep
    for t in range(n_timesteps):
        for f in range(n_features):
            feature_name = f"Timestep_{t}_Feature_{f}"
            scores = []
            
            # Repeat permutation
            for _ in range(n_repeats):
                # Create a copy of the data
                X_permuted = X.copy()
                
                # Permute the feature at the specific timestep
                permuted_idx = np.random.permutation(len(X))
                X_permuted[:, t, f] = X_permuted[permuted_idx, t, f]
                
                # Calculate score with permuted feature
                preds_permuted = model.predict(X_permuted)
                if metric == 'auc':
                    permuted_score = roc_auc_score(y, preds_permuted)
                else:
                    permuted_score = accuracy_score(y, (preds_permuted > 0.5).astype(int))
                
                # Calculate importance as decrease in performance
                importance = baseline_score - permuted_score
                scores.append(importance)
            
            # Average importance over repeats
            importance_scores[feature_name] = np.mean(scores)
    
    return importance_scores

def visualize_permutation_importance(importance_scores, n_timesteps=3, n_features=14):
    """
    Visualize permutation feature importance for LSTM input.
    
    Args:
        importance_scores (dict): Dictionary with importance scores
        n_timesteps (int): Number of timesteps
        n_features (int): Number of features
    """
    # Reshape scores for visualization
    importance_matrix = np.zeros((n_timesteps, n_features))
    
    for feature_name, score in importance_scores.items():
        t, f = map(int, feature_name.replace('Timestep_', '').replace('Feature_', '').split('_'))
        importance_matrix[t, f] = score
    
    plt.figure(figsize=(12, 7))
    im = plt.imshow(importance_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Importance (decrease in performance)')
    
    plt.yticks(range(n_timesteps), [f'Timestep {i}' for i in range(n_timesteps)])
    plt.xticks(range(n_features), [f'Feature {i}' for i in range(n_features)], rotation=45, ha='right')
    
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    
    # Also create a sorted bar chart for overall importance
    flat_importance = [(k, v) for k, v in importance_scores.items()]
    flat_importance.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 8))
    plt.bar([item[0] for item in flat_importance], [item[1] for item in flat_importance])
    plt.xticks(rotation=90)
    plt.title('Feature Importance (All Features Across Timesteps)')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()
    return plt

# Example usage
model = load_model('../saved_models/lstm_model_GTPU.h5')  # load the model
X_flattened = pd.read_csv("cumulative_X_test.csv")  # load data
X = X_flattened.values.reshape(-1, 3, 14)
y = pd.read_csv("cumulative_y_test.csv")

importance_scores = permutation_feature_importance(model, X, y)
visualize_permutation_importance(importance_scores)
