import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state
from tensorflow.keras.models import load_model
import pandas as pd

class LimeTimeSeriesExplainer:
    """
    LIME explainer for time series data used in LSTM models.
    """
    
    def __init__(self, kernel_width=0.75, random_state=None):
        """
        Initialize the explainer.
        
        Args:
            kernel_width (float): Width parameter for the exponential kernel
            random_state (int): Random seed for reproducible results
        """
        self.kernel_width = kernel_width
        self.random_state = check_random_state(random_state)
    
    def _generate_perturbations(self, x, num_samples=1000):
        """
        Generate perturbed samples around the input x.
        
        Args:
            x (np.ndarray): Original input with shape (3, 14)
            num_samples (int): Number of samples to generate
            
        Returns:
            tuple: (perturbed samples, binary features)
        """
        n_timesteps, n_features = x.shape
        
        # Generate binary features (1 = keep original, 0 = perturb)
        binary_features = self.random_state.randint(0, 2, size=(num_samples, n_timesteps, n_features))
        
        # Always include the original instance
        binary_features[0] = np.ones((n_timesteps, n_features))
        
        # Generate perturbed samples
        perturbed_samples = np.zeros((num_samples, n_timesteps, n_features))
        
        for i in range(num_samples):
            # For values to keep (binary_features == 1), use the original value
            # For values to perturb (binary_features == 0), use random noise
            mask = binary_features[i]
            perturbed_samples[i] = mask * x
            
            # Generate noise for perturbed features
            noise = self.random_state.normal(0, 0.1, size=(n_timesteps, n_features))
            perturbed_samples[i] += (1 - mask) * noise
        
        return perturbed_samples, binary_features
    
    def _calculate_weights(self, binary_features):
        """
        Calculate sample weights based on distance from original.
        
        Args:
            binary_features (np.ndarray): Binary features with shape (num_samples, n_timesteps, n_features)
            
        Returns:
            np.ndarray: Sample weights
        """
        # Calculate distance (number of perturbed features)
        distances = np.sum(1 - binary_features, axis=(1, 2))
        
        # Apply kernel function
        weights = np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))
        
        return weights
    
    def explain_instance(self, model, x, num_samples=1000, num_features=None):
        """
        Explain a single instance prediction.
        
        Args:
            model (tf.keras.Model): Trained LSTM model
            x (np.ndarray): Input to explain with shape (3, 14)
            num_samples (int): Number of samples to generate
            num_features (int): Number of top features to return
            
        Returns:
            dict: Feature importance dictionary
        """
        # Reshape x for single instance if needed
        x_orig_shape = x.shape
        if len(x_orig_shape) == 3 and x_orig_shape[0] == 1:  # (1, 3, 14)
            x = x[0]
        
        # Generate perturbed samples and binary features
        perturbed_samples, binary_features = self._generate_perturbations(x, num_samples)
        
        # Get model predictions for perturbed samples
        perturbed_shape = perturbed_samples.shape
        reshaped_samples = perturbed_samples.reshape(perturbed_shape[0], 1, perturbed_shape[1], perturbed_shape[2])
        predictions = model.predict(perturbed_samples.reshape(perturbed_shape[0], perturbed_shape[1], perturbed_shape[2]))
        
        # Calculate sample weights
        weights = self._calculate_weights(binary_features)
        
        # Fit weighted linear model
        n_timesteps, n_features = x.shape
        flattened_binary = binary_features.reshape(num_samples, -1)  # Flatten timesteps and features
        
        # Fit Ridge regression
        model_regressor = Ridge(alpha=1.0, fit_intercept=True)
        model_regressor.fit(flattened_binary, predictions.ravel(), sample_weight=weights)
        
        # Get feature importances
        importances = model_regressor.coef_.reshape(n_timesteps, n_features)
        intercept = model_regressor.intercept_
        
        # Get R-squared to evaluate explanation quality
        y_pred = model_regressor.predict(flattened_binary)
        r2 = r2_score(predictions, y_pred, sample_weight=weights)
        
        # Create importance dictionary
        importance_dict = {}
        for t in range(n_timesteps):
            for f in range(n_features):
                feature_name = f"Timestep_{t}_Feature_{f}"
                importance_dict[feature_name] = importances[t, f]
        
        # Sort by absolute importance
        sorted_importances = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Limit to top features if requested
        if num_features is not None:
            sorted_importances = sorted_importances[:num_features]
        
        result = {
            'importances': dict(sorted_importances),
            'intercept': intercept,
            'r2': r2
        }
        
        return result
    
    def visualize_explanation(self, explanation, n_timesteps=3, n_features=14, feature_names=None):
        """
        Visualize LIME explanation for time series LSTM.
        
        Args:
            explanation (dict): Explanation from explain_instance
            n_timesteps (int): Number of timesteps
            n_features (int): Number of features
            feature_names (list): List of feature names
        """
        importances = explanation['importances']
        r2 = explanation['r2']
        
        # Reshape for heatmap
        importance_matrix = np.zeros((n_timesteps, n_features))
        
        for feature_name, score in importances.items():
            if '_Feature_' in feature_name:
                t, f = map(int, feature_name.replace('Timestep_', '').replace('Feature_', '').split('_'))
                importance_matrix[t, f] = score
        
        # Plot heatmap
        plt.figure(figsize=(12, 7))
        im = plt.imshow(importance_matrix, cmap='seismic', interpolation='nearest')
        plt.colorbar(im, label='Feature Importance')
        
        plt.yticks(range(n_timesteps), [f'Timestep {i}' for i in range(n_timesteps)])
        if feature_names is None:
            plt.xticks(range(n_features), [f'Feature {i}' for i in range(n_features)], rotation=45, ha='right')
        else:
            plt.xticks(range(n_features), feature_names, rotation=45, ha='right')
        
        plt.title(f'LIME Explanation (R² = {r2:.3f})')
        plt.tight_layout()
        
        # Bar chart of top features
        top_k = min(10, len(importances))
        top_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        plt.figure(figsize=(10, 6))
        feature_names = [item[0] for item in top_features]
        feature_importance = [item[1] for item in top_features]
        colors = ['green' if imp > 0 else 'red' for imp in feature_importance]
        
        plt.bar(feature_names, feature_importance, color=colors)
        plt.xticks(rotation=90)
        plt.title(f'Top {top_k} Features by LIME Importance')
        plt.ylabel('Feature Importance (Coefficient)')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()
        return plt

# Example usage
model = load_model('../saved_models/lstm_model_GTPU.h5')  # load the model
X_flattened = pd.read_csv("cumulative_X_test.csv")  # load data
X = X_flattened.values.reshape(-1, 3, 14)
y = pd.read_csv("cumulative_y_test.csv")

lime_explainer = LimeTimeSeriesExplainer()
explanation = lime_explainer.explain_instance(model, X[0])
lime_explainer.visualize_explanation(explanation)
