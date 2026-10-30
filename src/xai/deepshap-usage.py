import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from deepshaplstm import DeepSHAPExplainer
import pandas as pd

# Main function to demonstrate DeepSHAP usage
def demonstrate_deepshap():
    model = load_model('../saved_models/lstm_model_GTPU.h5')  # load the model
    X_flattened = pd.read_csv("cumulative_X_test.csv")  # load data
    X = X_flattened.values.reshape(-1, 3, 14)
    y = pd.read_csv("cumulative_y_test.csv")
    print("Debugging", X.shape)
    # Select a sample to explain
    sample_idx = 0
    X_sample = X[sample_idx:sample_idx+1]
    
    # Create DeepSHAP explainer with background data
    explainer = DeepSHAPExplainer(model, X[:20])  # Use 20 background samples
    print("After explainer")
    # Compute SHAP values
    shap_result = explainer.explain(X_sample)
    print("After shap_result")
    # Create feature and timestep names (replace with your actual feature names)
    feature_names = [f'Feature {i}' for i in range(X.shape[2])]
    timestep_names = [f'T-{i}' for i in range(X.shape[1])]
    
    # Visualize SHAP values
    explainer.visualize_shap_values(shap_result, feature_names, timestep_names)
    
    # Create waterfall plot
    explainer.plot_shap_waterfall(shap_result, top_k=10, feature_names=feature_names, timestep_names=timestep_names)
    
    # Show plots
    plt.show()
    
    return shap_result


# Compare with another XAI method (e.g., Integrated Gradients)
def compare_xai_methods(model, X_sample):
    """Compare DeepSHAP with Integrated Gradients."""
    
    # DeepSHAP
    explainer = DeepSHAPExplainer(model, X_train[:20])
    shap_result = explainer.explain(X_sample)
    
    # Integrated Gradients (assuming you have the function from previous code)
    from integrated_gradients import integrated_gradients
    ig_attributions = integrated_gradients(model, X_sample)
    
    # Create comparison plots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # SHAP heatmap
    im1 = ax[0].imshow(shap_result['shap_values'], cmap='seismic', interpolation='nearest')
    ax[0].set_title('DeepSHAP Values')
    ax[0].set_yticks(range(X_sample.shape[1]))
    ax[0].set_yticklabels([f'Timestep {i}' for i in range(X_sample.shape[1])])
    ax[0].set_xticks(range(X_sample.shape[2]))
    ax[0].set_xticklabels([f'Feature {i}' for i in range(X_sample.shape[2])], rotation=90)
    plt.colorbar(im1, ax=ax[0])
    
    # IG heatmap
    im2 = ax[1].imshow(ig_attributions, cmap='seismic', interpolation='nearest')
    ax[1].set_title('Integrated Gradients')
    ax[1].set_yticks(range(X_sample.shape[1]))
    ax[1].set_yticklabels([f'Timestep {i}' for i in range(X_sample.shape[1])])
    ax[1].set_xticks(range(X_sample.shape[2]))
    ax[1].set_xticklabels([f'Feature {i}' for i in range(X_sample.shape[2])], rotation=90)
    plt.colorbar(im2, ax=ax[1])
    
    plt.tight_layout()
    plt.show()


# Run demo if executed as script
if __name__ == "__main__":
    demonstrate_deepshap()
