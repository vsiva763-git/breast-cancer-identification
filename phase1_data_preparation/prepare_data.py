"""Phase 1: Data Preparation - Main pipeline."""

import os
import yaml
from pathlib import Path
from utils.data_utils import (
    DataAugmenter, 
    BreakHisLoader, 
    CBISDDSMLoader,
    create_balanced_splits
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_breakhis_dataset(config: dict, data_path: str):
    """Prepare BreakHis dataset."""
    print("Loading BreakHis dataset...")
    loader = BreakHisLoader(data_path)
    images, labels = loader.load_dataset()
    
    print(f"Total samples: {len(images)}")
    print(f"Benign: {sum(1 for l in labels if l == 0)}")
    print(f"Malignant: {sum(1 for l in labels if l == 1)}")
    
    # Create balanced splits
    splits = create_balanced_splits(
        images, 
        labels,
        train_ratio=config['data']['splits']['train'],
        val_ratio=config['data']['splits']['val']
    )
    
    return {'images': images, 'labels': labels, 'splits': splits}


def prepare_cbis_ddsm_dataset(config: dict, data_path: str):
    """Prepare CBIS-DDSM dataset."""
    print("Loading CBIS-DDSM dataset...")
    loader = CBISDDSMLoader(data_path)
    images, labels = loader.load_dataset()
    
    print(f"Total samples: {len(images)}")
    print(f"Benign: {sum(1 for l in labels if l == 0)}")
    print(f"Malignant: {sum(1 for l in labels if l == 1)}")
    
    # Create balanced splits
    splits = create_balanced_splits(
        images, 
        labels,
        train_ratio=config['data']['splits']['train'],
        val_ratio=config['data']['splits']['val']
    )
    
    return {'images': images, 'labels': labels, 'splits': splits}


def main():
    """Main data preparation pipeline."""
    config = load_config('configs/config.yaml')
    
    print("=" * 60)
    print("Phase 1: Data Preparation")
    print("=" * 60)
    
    # Initialize augmenter
    augmenter = DataAugmenter(config['data']['augmentation'])
    
    print(f"\nData Augmentation Enabled: {config['data']['augmentation']['enabled']}")
    print(f"Train/Val/Test Split: {config['data']['splits']['train']}/{config['data']['splits']['val']}/0.15")
    
    # Add paths to datasets as needed
    # breakhis_data = prepare_breakhis_dataset(config, 'data/BreakHis')
    # cbis_ddsm_data = prepare_cbis_ddsm_dataset(config, 'data/CBIS-DDSM')
    
    print("\n✓ Data preparation pipeline initialized")
    print("  - DataAugmenter ready")
    print("  - Dataset loaders ready")
    print("  - Balanced split strategy configured")


if __name__ == '__main__':
    main()
