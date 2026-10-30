import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DeepSHAPExplainer:
    """
    Custom DeepSHAP implementation for TensorFlow LSTM models.
    This approximates SHAP values without depending on the external SHAP library.
    """
    
    def __init__(self, model, background_data=None):
        """
        Initialize the DeepSHAP explainer.
        
        Args:
            model (tf.keras.Model): Trained LSTM model
            background_data (np.ndarray): Background data for reference values
        """
        self.model = model
        self.background_data = background_data
    
    def _get_reference_values(self, n_samples=20):
        """
        Get reference values from background data or zeros.
        
        Args:
            n_samples (int): Number of samples to use from background data
            
        Returns:
            np.ndarray: Reference values
        """
        if self.background_data is not None:
            # Use randomly selected background data points
            indices = np.random.choice(len(self.background_data), size=min(n_samples, len(self.background_data)), replace=False)
            return self.background_data[indices]
        else:
            # Use zeros as reference values
            # Assuming input shape (?, 3, 14) - use single zero reference
            return np.zeros((1, 3, 14))

    def _interpolate_inputs(self, x, reference, alphas):
        print("Inside _interpolate_inputs")
        """
        Create interpolated inputs between reference and x.

        Args:
            x (np.ndarray): Input data point (3, 14)
            reference (np.ndarray): Reference values (n_ref, 3, 14)
            alphas (np.ndarray): Alpha values for interpolation

        Returns:
            tuple: (interpolated inputs, alphas repeated for each reference value)
        """
        n_refs = reference.shape[0]
        n_alphas = len(alphas)

        # Prepare outputs array
        interps = []

        # For each reference, create interpolations with all alphas
        for i in range(n_refs):
            # Calculate x - reference for this reference point
            diff = x - reference[i]  # Shape: (3, 14)

            # For each alpha, calculate reference + alpha * (x - reference)
            for alpha in alphas:
                interp = reference[i] + alpha * diff  # Shape: (3, 14)
                interps.append(interp)

        # Stack along first dimension
        interpolated = np.stack(interps)  # Shape: (n_refs * n_alphas, 3, 14)

        # For the alphas, we need [alpha1, alpha2, ..., alphan] repeated for each reference
        # For n_refs=2, n_alphas=3: [a1, a2, a3, a1, a2, a3]
        alphas_repeated = np.tile(alphas, n_refs)

        return interpolated, alphas_repeated
    
    def compute_gradients(self, x, reference_values):
        """
        Compute gradients for DeepSHAP.
        
        Args:
            x (np.ndarray): Input sample
            reference_values (np.ndarray): Reference values
            
        Returns:
            tuple: (gradients, predictions, reference_preds)
        """
        print("Shape of x at the start of the method: ", x.shape)
        # Number of steps for the integration approximation
        num_steps = 50
        
        # Generate alpha values for interpolation
        alphas = np.linspace(0, 1, num_steps)
        
        # Generate interpolated inputs
        inputs, _ = self._interpolate_inputs(x, reference_values, alphas)
        print("Inputs shape after the interpolation: ",inputs.shape)
        # Convert to tensor
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        print("Input tensor shape: ", inputs_tensor.shape)

        # Get gradients using GradientTape
        with tf.GradientTape() as tape:
            tape.watch(inputs_tensor)
            predictions = self.model(inputs_tensor)
            
        gradients = tape.gradient(predictions, inputs_tensor)
        
        # Get predictions
        predictions = predictions.numpy().flatten()
        
        # Get reference predictions (at alpha=0)
        reference_preds = self.model.predict(reference_values).flatten()
        
        return gradients.numpy(), predictions, reference_preds
    
    def explain(self, x, n_samples=20):
        """
        Compute DeepSHAP values for an input sample.
        
        Args:
            x (np.ndarray): Input sample with shape (1, 3, 14)
            n_samples (int): Number of background samples to use
            
        Returns:
            dict: Dictionary with SHAP values and related information
        """
        # Make sure x has the right shape
        if len(x.shape) == 2:  # If (3, 14)
            x = np.expand_dims(x, axis=0)  # Make it (1, 3, 14)
        
        # Get reference values
        reference_values = self._get_reference_values(n_samples)
        
        # Compute gradients
        gradients, predictions, reference_preds = self.compute_gradients(x[0], reference_values)
        
        # Average gradients across reference samples
        avg_gradients = np.mean(gradients.reshape(len(reference_values), -1, *x.shape[1:]), axis=0)
        
        # Compute baseline difference
        input_diff = x - np.mean(reference_values, axis=0)
        
        # Compute DeepSHAP values
        shap_values = avg_gradients * input_diff
        
        # Original prediction value
        original_pred = self.model.predict(x).flatten()[0]
        
        # Baseline value (average of reference predictions)
        baseline_value = np.mean(reference_preds)
        
        return {
            'shap_values': shap_values[0],  # Shape (3, 14)
            'expected_value': baseline_value,
            'prediction': original_pred,
            'reference_preds': reference_preds
        }
    
    def visualize_shap_values(self, shap_result, feature_names=None, timestep_names=None):
        """
        Visualize SHAP values for LSTM input.
        
        Args:
            shap_result (dict): Result from explain method
            feature_names (list): List of feature names
            timestep_names (list): List of timestep names
        """
        shap_values = shap_result['shap_values']
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(shap_values.shape[1])]
        
        if timestep_names is None:
            timestep_names = [f'Timestep {i}' for i in range(shap_values.shape[0])]
        
        # 1. Heatmap visualization
        plt.figure(figsize=(12, 7))
        
        # Get absolute max for symmetric colormap
        max_val = np.max(np.abs(shap_values))
        
        im = plt.imshow(shap_values, cmap='seismic', vmin=-max_val, vmax=max_val, interpolation='nearest')
        plt.colorbar(im, label='SHAP Value')
        
        plt.yticks(range(len(timestep_names)), timestep_names)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        
        plt.title(f'DeepSHAP Values (Prediction: {shap_result["prediction"]:.4f})')
        plt.tight_layout()
        
        # 2. Bar chart of most important features
        # Flatten the SHAP values
        flat_shap = []
        for t in range(shap_values.shape[0]):
            for f in range(shap_values.shape[1]):
                name = f"{timestep_names[t]}_{feature_names[f]}"
                value = shap_values[t, f]
                flat_shap.append((name, value))
        
        # Sort by absolute importance
        flat_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 10 features
        top_k = min(10, len(flat_shap))
        top_features = flat_shap[:top_k]
        
        plt.figure(figsize=(10, 6))
        
        feature_names = [item[0] for item in top_features]
        feature_values = [item[1] for item in top_features]
        colors = ['green' if x > 0 else 'red' for x in feature_values]
        
        plt.bar(feature_names, feature_values, color=colors)
        plt.xticks(rotation=90)
        plt.title(f'Top {top_k} Features by SHAP Value')
        plt.ylabel('SHAP Value')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # 3. Summary plot - show impact per timestep
        plt.figure(figsize=(8, 6))
        
        timestep_importance = [np.sum(np.abs(shap_values[i])) for i in range(len(timestep_names))]
        
        plt.bar(timestep_names, timestep_importance)
        plt.title('Feature Importance by Timestep')
        plt.ylabel('Total SHAP Impact (absolute sum)')
        plt.tight_layout()
        
        return plt

    def plot_shap_waterfall(self, shap_result, top_k=10, feature_names=None, timestep_names=None):
        """
        Create a waterfall plot of SHAP values.
        
        Args:
            shap_result (dict): Result from explain method
            top_k (int): Number of top features to show
            feature_names (list): List of feature names
            timestep_names (list): List of timestep names
        """
        shap_values = shap_result['shap_values']
        expected_value = shap_result['expected_value']
        prediction = shap_result['prediction']
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(shap_values.shape[1])]
        
        if timestep_names is None:
            timestep_names = [f'Timestep {i}' for i in range(shap_values.shape[0])]
        
        # Flatten SHAP values
        flat_shap = []
        for t in range(shap_values.shape[0]):
            for f in range(shap_values.shape[1]):
                name = f"{timestep_names[t]}_{feature_names[f]}"
                value = shap_values[t, f]
                flat_shap.append((name, value))
        
        # Sort by absolute value
        flat_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top features
        top_features = flat_shap[:top_k]
        
        # Create lists for plotting
        features = [item[0] for item in top_features]
        values = [item[1] for item in top_features]
        
        # Create the waterfall plot
        plt.figure(figsize=(12, 8))
        
        # Start with baseline value
        y_pos = np.arange(len(features) + 2)
        plot_values = [0] * (len(features) + 2)
        plot_values[0] = expected_value
        plot_values[-1] = prediction
        
        # Calculate cumulative sums for waterfall
        running_total = expected_value
        for i, val in enumerate(values):
            running_total += val
            plot_values[i+1] = val
        
        # Feature names with + or - indicators
        feature_labels = [f"{f} ({'+' if v > 0 else ''}{v:.4f})" for f, v in zip(features, values)]
        labels = ['Base value'] + feature_labels + ['Final prediction']
        
        # Colors based on positive/negative
        colors = ['gray']
        for val in values:
            colors.append('green' if val > 0 else 'red')
        colors.append('blue')
        
        # Create the plot
        plt.bar(y_pos, plot_values, color=colors)
        plt.axhline(y=expected_value, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=prediction, color='blue', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate([expected_value] + values + [prediction]):
            if i == 0:
                plt.text(i, expected_value + 0.01, f"{expected_value:.4f}", ha='center')
            elif i == len(y_pos) - 1:
                plt.text(i, prediction + 0.01, f"{prediction:.4f}", ha='center')
            else:
                # Skip small values to prevent overcrowding
                if abs(v) > 0.01:
                    y_offset = 0.01 if v > 0 else -0.03
                    plt.text(i, v + y_offset, f"{v:.4f}", ha='center')
        
        plt.xticks(y_pos, labels, rotation=90)
        plt.title('SHAP Value Waterfall Plot')
        plt.ylabel('Contribution to prediction')
        plt.tight_layout()
        
        return plt
