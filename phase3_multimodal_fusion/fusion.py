"""Phase 3: Multi-Modal Fusion - Fusion strategies."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for multi-modal learning."""
    
    def __init__(self, feature_dim: int = 512, num_modalities: int = 2):
        super().__init__()
        self.num_modalities = num_modalities
        
        # Attention weights for each modality
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, 256),
            nn.ReLU(),
            nn.Linear(256, num_modalities),
            nn.Softmax(dim=1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
    
    def forward(self, features_list: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention fusion.
        
        Args:
            features_list: List of feature tensors from different modalities
        
        Returns:
            fused_features: Fused feature representation
            attention_weights: Attention weights for each modality
        """
        # Concatenate features
        concatenated = torch.cat(features_list, dim=1)
        
        # Compute attention weights
        attention_weights = self.attention(concatenated)
        
        # Apply attention weights to each modality
        weighted_features = [
            features_list[i] * attention_weights[:, i:i+1] 
            for i in range(self.num_modalities)
        ]
        
        # Fuse weighted features
        weighted_concatenated = torch.cat(weighted_features, dim=1)
        fused_features = self.fusion(weighted_concatenated)
        
        return fused_features, attention_weights


class EarlyFusion(nn.Module):
    """Early fusion - concatenate features before processing."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
    
    def forward(self, features_list: list) -> torch.Tensor:
        """Concatenate and process features."""
        concatenated = torch.cat(features_list, dim=1)
        fused_features = self.fusion(concatenated)
        return fused_features


class LateFusion(nn.Module):
    """Late fusion - average classification outputs."""
    
    def forward(self, logits_list: list) -> torch.Tensor:
        """Average logits from different models."""
        return torch.mean(torch.stack(logits_list), dim=0)


class MultiModalClassifier(nn.Module):
    """Multi-modal classifier with fusion."""
    
    def __init__(
        self,
        histo_model: nn.Module,
        mammo_model: nn.Module,
        fusion_strategy: str = 'attention_based',
        num_classes: int = 2
    ):
        super().__init__()
        self.histo_model = histo_model
        self.mammo_model = mammo_model
        
        if fusion_strategy == 'attention_based':
            self.fusion = AttentionFusion(feature_dim=256, num_modalities=2)
        elif fusion_strategy == 'early':
            self.fusion = EarlyFusion(feature_dim=256)
        elif fusion_strategy == 'late':
            self.fusion = LateFusion()
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(
        self, 
        histo_images: torch.Tensor, 
        mammo_images: torch.Tensor
    ) -> Dict:
        """
        Forward pass for multi-modal input.
        
        Args:
            histo_images: Histopathology images
            mammo_images: Mammography images
        
        Returns:
            Dictionary with predictions and intermediate outputs
        """
        # Extract features from each modality
        histo_features, histo_logits = self.histo_model(histo_images)
        mammo_features, mammo_logits = self.mammo_model(mammo_images)
        
        # Fuse features
        if isinstance(self.fusion, AttentionFusion):
            fused_features, attention_weights = self.fusion([histo_features, mammo_features])
        elif isinstance(self.fusion, EarlyFusion):
            fused_features = self.fusion([histo_features, mammo_features])
            attention_weights = None
        else:
            # Late fusion
            fused_logits = self.fusion([histo_logits, mammo_logits])
            return {'logits': fused_logits}
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'histo_logits': histo_logits,
            'mammo_logits': mammo_logits,
            'histo_features': histo_features,
            'mammo_features': mammo_features,
            'fused_features': fused_features,
            'attention_weights': attention_weights
        }


class EnsembleVoting(nn.Module):
    """Ensemble voting for multiple models."""
    
    def __init__(self, models: list, voting_type: str = 'majority'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting_type = voting_type
    
    def forward(self, *inputs) -> torch.Tensor:
        """Ensemble voting."""
        predictions = []
        
        for model in self.models:
            output = model(*inputs)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            predictions.append(logits)
        
        if self.voting_type == 'majority':
            # Majority voting on predicted classes
            predictions = torch.stack([torch.argmax(p, dim=1) for p in predictions])
            mode_result = torch.mode(predictions, dim=0)
            return mode_result.values
        else:
            # Average logits
            return torch.mean(torch.stack(predictions), dim=0)


def main():
    """Simple entrypoint to verify fusion module configuration."""
    import yaml
    try:
        config = yaml.safe_load(open('configs/config.yaml'))
        strategy = config.get('fusion', {}).get('strategy', 'attention_based')
        print("=" * 60)
        print("Phase 3: Multi-Modal Fusion")
        print("=" * 60)
        print(f"\nFusion strategy: {strategy}")
        print("\n✓ Fusion module is ready")
    except Exception as e:
        print(f"✗ Fusion module initialization failed: {e}")


if __name__ == '__main__':
    main()
