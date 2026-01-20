#!/usr/bin/env python3
"""Download and extract BreakHis dataset."""

import os
import urllib.request
import tarfile
from pathlib import Path
import sys

# Configuration
DATASET_URL = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
DATA_DIR = Path("data")
DOWNLOAD_PATH = DATA_DIR / "BreaKHis_v1.tar.gz"
EXTRACT_DIR = DATA_DIR

def download_dataset():
    """Download BreakHis dataset from UFPR server."""
    
    print("=" * 70)
    print("BreakHis Dataset Downloader")
    print("=" * 70)
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Check if already downloaded
    if DOWNLOAD_PATH.exists():
        print(f"✓ Dataset already downloaded at {DOWNLOAD_PATH}")
        response = input("Extract it? (y/n): ").lower()
        if response != 'y':
            return
    else:
        print(f"\n📥 Downloading BreakHis dataset...")
        print(f"URL: {DATASET_URL}")
        print(f"Destination: {DOWNLOAD_PATH}")
        print(f"Size: ~1.2 GB (this may take 10-30 minutes)\n")
        
        try:
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 // total_size, 100)
                bar_length = 50
                filled = int(bar_length * downloaded // total_size)
                bar = '█' * filled + '░' * (bar_length - filled)
                sys.stdout.write(f'\r[{bar}] {percent}% ({downloaded/1e9:.1f}GB/{total_size/1e9:.1f}GB)')
                sys.stdout.flush()
            
            urllib.request.urlretrieve(DATASET_URL, DOWNLOAD_PATH, download_progress)
            print("\n✓ Download complete!")
            
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            print("\nAlternative: Download manually from:")
            print("https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/")
            return False
    
    # Extract dataset
    print(f"\n📦 Extracting dataset to {EXTRACT_DIR}...")
    try:
        with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
        print("✓ Extraction complete!")
        
        # Verify structure
        breakhis_dir = EXTRACT_DIR / "BreaKHis_v1"
        if breakhis_dir.exists():
            print(f"✓ Dataset extracted to: {breakhis_dir}")
            
            # Count images
            benign_count = len(list((breakhis_dir / "benign").rglob("*.png")))
            malignant_count = len(list((breakhis_dir / "malignant").rglob("*.png")))
            print(f"\nDataset Statistics:")
            print(f"  Benign images: {benign_count}")
            print(f"  Malignant images: {malignant_count}")
            print(f"  Total images: {benign_count + malignant_count}")
            
            return True
        else:
            print("✗ Extraction verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

if __name__ == "__main__":
    success = download_dataset()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ BreakHis dataset ready!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run: python phase1_data_preparation/prepare_data.py")
        print("2. Or open: notebooks/Phase1_DataPreparation_Colab.ipynb")
    else:
        print("\n" + "=" * 70)
        print("⚠️  Dataset download failed")
        print("=" * 70)
        sys.exit(1)
