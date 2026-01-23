#!/usr/bin/env python3
"""Non-interactive BreakHis downloader with resume and extraction."""
from pathlib import Path
import tarfile
import sys

# Import functions from download_breakhis
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_breakhis import DATASET_URL, DATA_DIR, DOWNLOAD_PATH, EXTRACT_DIR, _download_with_resume

print("Starting non-interactive BreakHis download...")
DATA_DIR.mkdir(exist_ok=True)

ok = _download_with_resume(DATASET_URL, DOWNLOAD_PATH)
if not ok:
    print("Download failed. Exiting with code 1.")
    sys.exit(1)

print("Download complete. Extracting archive...")
try:
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)
except Exception as e:
    print(f"Extraction failed: {e}")
    sys.exit(1)

breakhis_dir = EXTRACT_DIR / "BreaKHis_v1"
if not breakhis_dir.exists():
    print("Extraction verification failed: BreaKHis_v1 not found.")
    sys.exit(1)

print(f"Success: Extracted to {breakhis_dir}")
