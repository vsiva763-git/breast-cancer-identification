#!/usr/bin/env python3
"""Quick training demo with fewer epochs for demonstration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
from tqdm import tqdm
import cv2
from datetime import datetime

from utils.model_utils import EfficientNetHistopathology
from utils.data_utils import BreakHisLoader, create_balanced_splits, DataAugmenter


class ImageDataset(Dataset):
    """Dataset with automatic resizing."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.FloatTensor(image) / 255.0
        
        return image, torch.LongTensor([label])[0]


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                _, logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(val_loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        _, logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(val_loader), correct / total


def main():
    print("\n" + "="*70)
    print(" QUICK TRAINING DEMO - 5 Epochs")
    print("="*70 + "\n")
    
    config = yaml.safe_load(open('configs/config.yaml'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 Device: {device}\n")
    
    # Load data
    print(f"📂 Loading dataset...")
    loader = BreakHisLoader('data/BreaKHis_v1')
    images, labels = loader.load_dataset()
    splits = create_balanced_splits(images, labels)
    augmenter = DataAugmenter(config['data']['augmentation'])
    
    # Create small datasets for quick demo
    train_indices = splits['train'][:2000]  # Use subset for speed
    val_indices = splits['val'][:400]
    
    train_dataset = ImageDataset(
        [images[i] for i in train_indices],
        [labels[i] for i in train_indices],
        transform=augmenter.train_transform
    )
    
    val_dataset = ImageDataset(
        [images[i] for i in val_indices],
        [labels[i] for i in val_indices],
        transform=augmenter.val_transform
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    
    # Initialize model
    print(f"🧠 Initializing EfficientNet-B0...")
    model = EfficientNetHistopathology(num_classes=2, dropout=0.3, pretrained=True).to(device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    # Train
    print(f"\n⚡ Starting training (5 epochs)...\n")
    start = datetime.now()
    
    for epoch in range(5):
        print(f"[Epoch {epoch+1}/5]")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"  Train: Loss={train_loss:.4f} Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} Acc={val_acc:.4f}\n")
    
    elapsed = (datetime.now() - start).total_seconds()
    
    print("="*70)
    print(f"✅ TRAINING COMPLETE in {elapsed:.1f} seconds")
    print(f"   Model is ready for Phase 3 Fusion and Phase 4 XAI!")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
