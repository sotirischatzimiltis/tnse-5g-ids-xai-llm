import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd


def integrated_gradients(model, input_sample, baseline=None, steps=50):
    """
    Compute integrated gradients for a given input sample.

    Args:
        model (tf.keras.Model): Trained LSTM model
        input_sample (np.ndarray): Input to explain, shape (1, 3, 14)
        baseline (np.ndarray): Baseline input, if None zeros are used
        steps (int): Number of steps for the integral approximation

    Returns:
        np.ndarray: Feature attributions with shape (3, 14)
    """
    # Remove batch dimension if it exists
    if input_sample.shape[0] == 1:
        input_sample = input_sample[0]  # Shape becomes (3, 14)

    # Create baseline (zeros) if not provided
    if baseline is None:
        baseline = np.zeros_like(input_sample)  # Shape (3, 14)

    # Cast to float32
    input_sample = tf.cast(input_sample, tf.float32)
    baseline = tf.cast(baseline, tf.float32)

    # Compute the difference between input and baseline
    difference = input_sample - baseline  # Shape (3, 14)

    # Generate interpolated inputs
    alphas = np.linspace(0, 1, steps)
    interpolated_inputs = []

    for alpha in alphas:
        interpolated = baseline + alpha * difference  # Shape (3, 14)
        interpolated_inputs.append(interpolated)

    # Stack with a batch dimension
    interpolated_inputs = tf.stack(interpolated_inputs)  # Shape (steps, 3, 14)

    # Compute gradients for each interpolated input
    with tf.GradientTape() as tape:
        # Watch the inputs
        tape.watch(interpolated_inputs)
        # Get predictions
        predictions = model(interpolated_inputs)

    # Get gradients
    gradients = tape.gradient(predictions, interpolated_inputs)

    # Compute integral approximation using trapezoidal rule
    avg_gradients = tf.reduce_mean(gradients, axis=0)  # Average over steps
    integrated_grads = avg_gradients * difference

    return integrated_grads.numpy()  # Return with shape (3, 14)

def visualize_integrated_gradients(attributions, feature_names=None, timestep_names=None):
    """
    Visualize integrated gradients attributions for LSTM input.
    
    Args:
        attributions (np.ndarray): Attributions with shape (3, 14)
        feature_names (list): List of feature names
        timestep_names (list): List of timestep names
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(attributions.shape[1])]
    
    if timestep_names is None:
        timestep_names = [f'Timestep {i}' for i in range(attributions.shape[0])]
    
    plt.figure(figsize=(12, 7))
    im = plt.imshow(attributions, cmap='seismic', interpolation='nearest')
    plt.colorbar(im, label='Attribution')
    
    plt.yticks(range(len(timestep_names)), timestep_names)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    
    plt.title('Integrated Gradients Attributions')
    plt.tight_layout()
    plt.show()
    return plt

# Example usage
model = load_model('../saved_models/lstm_model_GTPU.h5')  # load the model
X_flattened = pd.read_csv("cumulative_X_test.csv")  # load data
X = X_flattened.values.reshape(-1, 3, 14)
y = pd.read_csv("cumulative_y_test.csv")

# Assuming model is your trained LSTM model and X_sample is a single input sample
X_sample = X[0:1]  # Shape (1, 3, 14)
attributions = integrated_gradients(model, X_sample)
visualize_integrated_gradients(attributions)
