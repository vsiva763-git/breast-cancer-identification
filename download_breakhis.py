#!/usr/bin/env python3
"""Download and extract BreakHis dataset."""

import os
import urllib.request
import tarfile
from pathlib import Path
import sys
import time
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

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
        existing_size = DOWNLOAD_PATH.stat().st_size
        remote_size = None
        if requests is not None:
            try:
                head = requests.head(DATASET_URL, timeout=30)
                if head.ok and 'Content-Length' in head.headers:
                    remote_size = int(head.headers['Content-Length'])
            except Exception:
                remote_size = None

        print(f"✓ Found existing archive at {DOWNLOAD_PATH} ({existing_size/1e9:.2f} GB)")
        incomplete = False
        if remote_size and existing_size < remote_size:
            incomplete = True
        # Heuristic fallback: dataset is ~4.3GB; anything <1GB is incomplete
        if not remote_size and existing_size < 1_000_000_000:
            incomplete = True
        if incomplete:
            print("\nThe archive appears incomplete.")
            resp = input("Resume download now? (y/n): ").lower()
            if resp == 'y':
                ok = _download_with_resume(DATASET_URL, DOWNLOAD_PATH)
                if not ok:
                    print("✗ Resume failed. Please try later or download manually.")
                    return False
            else:
                print("Skipping resume.")

        response = input("Extract it? (y/n): ").lower()
        if response != 'y':
            return
    else:
        print(f"\n📥 Downloading BreakHis dataset...")
        print(f"URL: {DATASET_URL}")
        print(f"Destination: {DOWNLOAD_PATH}")
        print(f"Note: ~4.3 GB download. Resume supported.\n")

        # Prefer requests with resume support if available
        try:
            success = False
            if requests is not None:
                success = _download_with_resume(DATASET_URL, DOWNLOAD_PATH)
            else:
                success = _download_with_urlretrieve(DATASET_URL, DOWNLOAD_PATH)

            if not success:
                print("\n✗ Download failed")
                print("\nAlternative: Download manually from:")
                print("https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/")
                return False

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


def _download_with_urlretrieve(url: str, dest: Path) -> bool:
    """Fallback downloader using urllib.urlretrieve (no resume)."""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(int(downloaded * 100 / total_size), 100) if total_size else 0
        bar_length = 50
        filled = int(bar_length * (percent / 100))
        bar = '█' * filled + '░' * (bar_length - filled)
        if total_size:
            sys.stdout.write(f'\r[{bar}] {percent}% ({downloaded/1e9:.1f}GB/{total_size/1e9:.1f}GB)')
        else:
            sys.stdout.write(f'\r[{bar}] {percent}% ({downloaded/1e9:.1f}GB/??GB)')
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, download_progress)
    return True


def _download_with_resume(url: str, dest: Path, max_retries: int = 5, chunk_mb: int = 8) -> bool:
    """Download with resume support using requests.

    Args:
        url: Source URL
        dest: Target file path
        max_retries: Number of retries on failure
        chunk_mb: Chunk size in MB

    Returns:
        True if download completed, False otherwise.
    """
    if requests is None:
        return _download_with_urlretrieve(url, dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try to get total size
    total_size: Optional[int] = None
    try:
        head = requests.head(url, timeout=30)
        if head.ok and 'Content-Length' in head.headers:
            total_size = int(head.headers['Content-Length'])
    except Exception:
        pass

    # Resume if partial exists
    downloaded = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if downloaded and total_size and downloaded < total_size:
        headers['Range'] = f'bytes={downloaded}-'

    attempt = 0
    chunk_size = chunk_mb * 1024 * 1024
    start_time = time.time()

    while attempt < max_retries:
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code in (200, 206):
                    mode = 'ab' if 'Range' in headers else 'wb'
                    with open(dest, mode) as f:
                        current = downloaded
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                current += len(chunk)
                                _print_progress(current, total_size, start_time)
                    # Verify size if known
                    if total_size and dest.stat().st_size < total_size:
                        raise IOError('Incomplete download detected')
                    sys.stdout.write('\n')
                    return True
                else:
                    raise IOError(f'HTTP {r.status_code}')
        except Exception as e:
            attempt += 1
            print(f"\nRetry {attempt}/{max_retries} after error: {e}")
            time.sleep(3 * attempt)

    return False


def _print_progress(current: int, total: Optional[int], start_time: float):
    """Print progress bar with speed and ETA."""
    bar_length = 50
    if total:
        percent = min(int(current * 100 / total), 100)
        filled = int(bar_length * (percent / 100))
        bar = '█' * filled + '░' * (bar_length - filled)
    else:
        percent = 0
        filled = int(bar_length * 0.1)
        bar = '█' * filled + '░' * (bar_length - filled)

    elapsed = max(time.time() - start_time, 1e-6)
    speed_mb_s = (current / 1024 / 1024) / elapsed
    if total:
        remaining_bytes = max(total - current, 0)
        eta_s = remaining_bytes / 1024 / 1024 / speed_mb_s if speed_mb_s > 0 else float('inf')
        eta_str = f"ETA: {int(eta_s // 60)}m {int(eta_s % 60)}s"
        sys.stdout.write(
            f"\r[{bar}] {percent}% ({current/1e9:.1f}GB/{total/1e9:.1f}GB) "
            f"{speed_mb_s:.2f} MB/s | {eta_str}"
        )
    else:
        sys.stdout.write(
            f"\r[{bar}] {current/1e9:.1f}GB downloaded {speed_mb_s:.2f} MB/s"
        )
    sys.stdout.flush()

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
