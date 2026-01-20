"""Data utilities for loading and processing medical imaging datasets."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAugmenter:
    """Handles data augmentation for training."""
    
    def __init__(self, augmentation_config: Dict):
        self.config = augmentation_config
        self.train_transform = self._build_train_transforms()
        self.val_transform = self._build_val_transforms()
    
    def _build_train_transforms(self):
        """Build augmentation pipeline for training."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ], is_check_shapes=False)
    
    def _build_val_transforms(self):
        """Build augmentation pipeline for validation/testing."""
        return A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])


class BreakHisLoader:
    """Loads BreakHis histopathology dataset."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
    
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """Load BreakHis dataset."""
        images = []
        labels = []
        
        class_map = {'benign': 0, 'malignant': 1}
        
        for class_name, class_label in class_map.items():
            class_path = self.root_path / class_name
            if class_path.exists():
                for img_file in class_path.glob('*.png'):
                    images.append(str(img_file))
                    labels.append(class_label)
        
        return images, labels


class CBISDDSMLoader:
    """Loads CBIS-DDSM mammography dataset."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
    
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """Load CBIS-DDSM dataset."""
        images = []
        labels = []
        
        # Typical structure: calc_case_description.csv
        csv_path = self.root_path / 'calc_case_description.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Process based on pathology column
            for idx, row in df.iterrows():
                label = 1 if row['pathology'] == 'malignant' else 0
                images.append(row['image_path'])
                labels.append(label)
        
        return images, labels


def create_balanced_splits(
    images: List[str], 
    labels: List[int], 
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict:
    """Create balanced train/val/test splits."""
    
    # Separate by class
    benign_indices = [i for i, l in enumerate(labels) if l == 0]
    malignant_indices = [i for i, l in enumerate(labels) if l == 1]
    
    # Shuffle
    np.random.shuffle(benign_indices)
    np.random.shuffle(malignant_indices)
    
    # Split each class
    benign_train_size = int(len(benign_indices) * train_ratio)
    benign_val_size = int(len(benign_indices) * val_ratio)
    
    malignant_train_size = int(len(malignant_indices) * train_ratio)
    malignant_val_size = int(len(malignant_indices) * val_ratio)
    
    # Combine splits
    train_indices = (benign_indices[:benign_train_size] + 
                    malignant_indices[:malignant_train_size])
    
    val_indices = (benign_indices[benign_train_size:benign_train_size + benign_val_size] +
                  malignant_indices[malignant_train_size:malignant_train_size + malignant_val_size])
    
    test_indices = (benign_indices[benign_train_size + benign_val_size:] +
                   malignant_indices[malignant_train_size + malignant_val_size:])
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
