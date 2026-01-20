"""Phase 2: Lightweight Model Development - Training pipeline."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from typing import Dict, Tuple
from utils.model_utils import (
    EfficientNetHistopathology,
    MobileNetMammography,
    save_checkpoint,
    load_checkpoint
)


class ModelTrainer:
    """Trainer for lightweight models with mixed precision support."""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler() if config['models']['training']['mixed_precision'] else None
    
    def initialize_models(self) -> Tuple[nn.Module, nn.Module]:
        """Initialize both modality models."""
        print("Initializing models...")
        
        histo_model = EfficientNetHistopathology(
            num_classes=2,
            dropout=self.config['models']['histopathology']['dropout'],
            pretrained=self.config['models']['histopathology']['pretrained']
        ).to(self.device)
        
        mammo_model = MobileNetMammography(
            num_classes=2,
            dropout=self.config['models']['mammography']['dropout'],
            pretrained=self.config['models']['mammography']['pretrained']
        ).to(self.device)
        
        print(f"✓ Histopathology Model: EfficientNet-B0")
        print(f"✓ Mammography Model: MobileNet-V3")
        
        return histo_model, mammo_model
    
    def setup_training(self, model: nn.Module):
        """Setup optimizer and scheduler."""
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['models']['training']['learning_rate']
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['models']['training']['epochs']
        )
        
        criterion = nn.CrossEntropyLoss()
        
        return optimizer, scheduler, criterion
    
    def train_epoch(self, model: nn.Module, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            if self.config['models']['training']['mixed_precision']:
                with torch.cuda.amp.autocast():
                    _, logits = model(images)
                    loss = criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                _, logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, model: nn.Module, val_loader, criterion):
        """Validate model."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            _, logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy


def main():
    """Main training pipeline."""
    config = yaml.safe_load(open('configs/config.yaml'))
    
    print("=" * 60)
    print("Phase 2: Lightweight Model Development")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    trainer = ModelTrainer(config, device)
    
    # Initialize models
    histo_model, mammo_model = trainer.initialize_models()
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {config['models']['training']['epochs']}")
    print(f"  - Batch Size: {config['models']['training']['batch_size']}")
    print(f"  - Learning Rate: {config['models']['training']['learning_rate']}")
    print(f"  - Mixed Precision: {config['models']['training']['mixed_precision']}")
    print(f"  - Early Stopping: {config['models']['training']['early_stopping']}")
    
    print("\n✓ Model training pipeline initialized")


if __name__ == '__main__':
    main()
