"""Phase 4: Explainability (XAI) - Interpretability methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List
from captum.attr import GradCAM, Saliency
import shap


class GradCAMVisualizer:
    """Grad-CAM visualization for model interpretability."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradcam = GradCAM(model, target_layer)
    
    def generate_heatmap(self, input_image: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        
        # Get attributions
        attributions = self.gradcam.attribute(
            input_image,
            target=target_class,
            relu_attributions=True
        )
        
        # Convert to numpy
        heatmap = attributions.detach().cpu().numpy()[0].mean(axis=0)
        
        # Normalize to 0-1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on original image."""
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to color (hot colormap)
        heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay
        if len(image.shape) == 2:
            image_3channel = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            image_3channel = (image * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(image_3channel, 1 - alpha, heatmap_color, alpha, 0)
        
        return overlay


class SaliencyVisualizer:
    """Saliency map visualization."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.saliency = Saliency(model)
    
    def generate_saliency_map(self, input_image: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate saliency map."""
        
        input_image.requires_grad = True
        
        attributions = self.saliency.attribute(input_image, target=target_class)
        
        # Get magnitude of gradients
        saliency_map = torch.max(attributions.abs(), dim=1)[0]
        saliency_map = saliency_map.detach().cpu().numpy()[0]
        
        # Normalize to 0-1
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map


class SHAPExplainer:
    """SHAP-based feature importance."""
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor):
        self.model = model
        # Create explainer
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def get_shap_values(self, input_data: torch.Tensor) -> np.ndarray:
        """Get SHAP values for input data."""
        shap_values = self.explainer.shap_values(input_data)
        return shap_values
    
    def plot_shap_summary(self, shap_values: np.ndarray, feature_names: List[str] = None):
        """Plot SHAP summary."""
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]
        
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)


class AttentionVisualizer:
    """Visualize attention weights from fusion layer."""
    
    @staticmethod
    def visualize_attention_weights(
        attention_weights: torch.Tensor,
        modality_names: List[str] = None
    ) -> dict:
        """Visualize attention weights."""
        
        if modality_names is None:
            modality_names = ['Histopathology', 'Mammography']
        
        weights = attention_weights.detach().cpu().numpy()[0]
        
        visualization = {
            'weights': dict(zip(modality_names, weights)),
            'dominant_modality': modality_names[np.argmax(weights)]
        }
        
        return visualization


class ExplainabilityReport:
    """Generate comprehensive explainability report."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradcam = None
        self.saliency = None
    
    def generate_report(
        self,
        input_image: torch.Tensor,
        prediction: int,
        image_name: str = "sample"
    ) -> dict:
        """Generate comprehensive report."""
        
        report = {
            'image_name': image_name,
            'prediction': prediction,
            'visualizations': {}
        }
        
        return report
    
    def save_report(self, report: dict, output_path: str):
        """Save report as JSON."""
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
