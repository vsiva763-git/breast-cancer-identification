"""Phase 1: Data Preparation - Main pipeline."""

import os
import sys
import yaml
from pathlib import Path
from utils.data_utils import (
    DataAugmenter, 
    BreakHisLoader,
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
    
    # Prepare datasets
    datasets = {}
    
    # BreakHis dataset
    breakhis_path = 'data/BreaKHis_v1'
    if Path(breakhis_path).exists():
        print(f"\n📂 Loading BreakHis dataset from {breakhis_path}...")
        breakhis_data = prepare_breakhis_dataset(config, breakhis_path)
        datasets['BreakHis'] = breakhis_data
        print(f"✓ BreakHis: {len(breakhis_data['images'])} images loaded")
    else:
        print(f"\n⚠️  BreakHis not found at {breakhis_path}")
        print("   Download with: python download_breakhis.py")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Data Preparation Summary")
    print("=" * 60)
    if datasets:
        for name, data in datasets.items():
            print(f"\n{name}:")
            print(f"  Total images: {len(data['images'])}")
            print(f"  Train: {len(data['splits']['train'])} | Val: {len(data['splits']['val'])} | Test: {len(data['splits']['test'])}")
    else:
        print("No datasets found. Please download BreakHis dataset:")
        print("  python download_breakhis.py")


if __name__ == '__main__':
    main()
