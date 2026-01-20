"""Model utilities for loading and managing models."""

import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple


class EfficientNetHistopathology(nn.Module):
    """EfficientNet model for histopathology images."""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        
        # Remove original classification layer
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both features and logits."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits


class MobileNetMammography(nn.Module):
    """MobileNetV3 model for mammography images."""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('mobilenet_v3_small', pretrained=pretrained)
        
        # Remove original classification layer
        in_features = self.backbone.classifier[1].in_features if hasattr(self.backbone.classifier, '__getitem__') else 1024
        self.backbone.classifier = nn.Identity()
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both features and logits."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits


class KnowledgeDistillationModel(nn.Module):
    """Teacher-student knowledge distillation framework."""
    
    def __init__(self, teacher: nn.Module, student: nn.Module, temperature: float = 4.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass for both teacher and student."""
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)
        
        return {
            'student_logits': student_logits,
            'teacher_logits': teacher_logits
        }


def load_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """Factory function to load models."""
    
    if model_name == 'efficientnet_b0_histo':
        return EfficientNetHistopathology(num_classes=num_classes)
    elif model_name == 'mobilenet_v3_mammo':
        return MobileNetMammography(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_checkpoint(model: nn.Module, optimizer, epoch: int, save_path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


def load_checkpoint(model: nn.Module, optimizer, checkpoint_path: str):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']
