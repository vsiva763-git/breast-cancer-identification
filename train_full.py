#!/usr/bin/env python3
"""Phase 2: Full Model Training with GPU Acceleration."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import cv2

from utils.model_utils import EfficientNetHistopathology, MobileNetMammography
from utils.data_utils import BreakHisLoader, create_balanced_splits, DataAugmenter


class ImageDataset(Dataset):
    """Custom dataset for medical images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Return black image if load fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize to 224x224
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Simple normalization
            image = torch.FloatTensor(image) / 255.0
        
        return image, torch.LongTensor([label])[0]


class ModelTrainer:
    """Trainer for models with mixed precision support."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Use modern AMP
        self.use_amp = config['models']['training'].get('mixed_precision', True)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
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
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, model, val_loader, criterion):
        """Validate model."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            _, logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total
        return avg_loss, avg_acc
    
    def train_model(self, model, train_loader, val_loader, model_name="model"):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['models']['training']['learning_rate']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['models']['training']['epochs']
        )
        
        criterion = nn.CrossEntropyLoss()
        epochs = self.config['models']['training']['epochs']
        patience = self.config['models']['training'].get('patience', 5)
        
        for epoch in range(epochs):
            print(f"\n[Epoch {epoch+1}/{epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Scheduler step
            scheduler.step()
            
            # Log
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = f"checkpoints/{model_name}_best.pth"
                Path("checkpoints").mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"✓ Saved best checkpoint: {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n⚠ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n{'='*60}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        print(f"{'='*60}\n")
        
        return model


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print(" PHASE 2: LIGHTWEIGHT MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load config
    config = yaml.safe_load(open('configs/config.yaml'))
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load dataset
    print(f"\n📂 Loading BreakHis dataset...")
    loader = BreakHisLoader('data/BreaKHis_v1')
    images, labels = loader.load_dataset()
    print(f"   Total images: {len(images)}")
    print(f"   Benign: {sum(1 for l in labels if l == 0)}")
    print(f"   Malignant: {sum(1 for l in labels if l == 1)}")
    
    # Create splits
    splits = create_balanced_splits(
        images, labels,
        train_ratio=config['data']['splits']['train'],
        val_ratio=config['data']['splits']['val']
    )
    
    # Prepare augmentation
    augmenter = DataAugmenter(config['data']['augmentation'])
    
    # Create datasets
    train_dataset = ImageDataset(
        [images[i] for i in splits['train']],
        [labels[i] for i in splits['train']],
        transform=augmenter.train_transform
    )
    
    val_dataset = ImageDataset(
        [images[i] for i in splits['val']],
        [labels[i] for i in splits['val']],
        transform=augmenter.val_transform
    )
    
    # Create dataloaders
    batch_size = config['models']['training']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"\n📊 Data Loaders:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Batch size: {batch_size}")
    
    # Initialize trainer
    trainer = ModelTrainer(config, device)
    
    # Initialize models
    print(f"\n🧠 Initializing models...")
    histo_model = EfficientNetHistopathology(
        num_classes=2,
        dropout=config['models']['histopathology']['dropout'],
        pretrained=config['models']['histopathology']['pretrained']
    ).to(device)
    
    mammo_model = MobileNetMammography(
        num_classes=2,
        dropout=config['models']['mammography']['dropout'],
        pretrained=config['models']['mammography']['pretrained']
    ).to(device)
    
    print(f"   ✓ Histopathology: EfficientNet-B0")
    print(f"   ✓ Mammography: MobileNetV3")
    
    # Training config info
    print(f"\n⚡ Training Configuration:")
    print(f"   Epochs: {config['models']['training']['epochs']}")
    print(f"   Learning rate: {config['models']['training']['learning_rate']}")
    print(f"   Mixed precision: {config['models']['training'].get('mixed_precision', False)}")
    print(f"   Early stopping patience: {config['models']['training'].get('patience', 5)}")
    
    # Train models
    start_time = datetime.now()
    
    histo_model = trainer.train_model(
        histo_model,
        train_loader,
        val_loader,
        model_name="histopathology"
    )
    
    mammo_model = trainer.train_model(
        mammo_model,
        train_loader,
        val_loader,
        model_name="mammography"
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Save training metrics
    metrics = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accs': trainer.val_accs,
        'best_val_acc': trainer.best_val_acc,
        'total_time_seconds': elapsed,
        'device': device
    }
    
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Checkpoints saved to: checkpoints/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
