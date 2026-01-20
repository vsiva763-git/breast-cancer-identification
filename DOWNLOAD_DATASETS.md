# 📥 Dataset Download Guide

## Quick Start

### Automatic Download (Recommended)

```bash
# From project root directory
python download_breakhis.py
```

This will:
1. ✅ Download BreakHis dataset (~1.2 GB)
2. ✅ Extract to `data/BreaKHis_v1/`
3. ✅ Verify dataset structure
4. ✅ Count images

### Manual Download

**BreakHis Dataset**
- URL: http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz
- Alternative: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/
- Size: ~1.2 GB
- Format: tar.gz

**Steps:**
1. Download the tar.gz file
2. Extract to `data/` folder:
   ```bash
   tar -xzf BreaKHis_v1.tar.gz -C data/
   ```
3. Verify structure:
   ```
   data/
   └── BreaKHis_v1/
       ├── benign/
       │   ├── SOB_B_A-14-22549.png
       │   ├── SOB_B_A-14-22550.png
       │   └── ... (~5,429 images)
       └── malignant/
           ├── SOB_M_MC-14-22563.png
           ├── SOB_M_MC-14-22564.png
           └── ... (~2,480 images)
   ```

---

## BreakHis Dataset Details

### Overview
- **Total Images**: 7,909
- **Benign**: 5,429
- **Malignant**: 2,480
- **Resolution**: 700×460 pixels
- **Magnifications**: 40×, 100×, 200×, 400×

### Class Distribution
```
Benign:
├── Adenosis
├── Fibroadenoma
├── Phyllodes tumor
└── Tubular adenoma

Malignant:
├── Ductal carcinoma
├── Lobular carcinoma
├── Mucinous carcinoma
└── Papillary carcinoma
```

### File Structure
- Binary classification (benign vs malignant)
- PNG format
- Organized in `benign/` and `malignant/` directories
- Filename pattern: `SOB_[B/M]_[type]-[ID].png`

---

## CBIS-DDSM Dataset (Optional)

### Download from Kaggle
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-dataset

# Extract
unzip cbis-ddsm-breast-cancer-dataset.zip -d data/CBIS-DDSM/
```

### Details
- **Total Images**: ~6,000
- **Size**: ~3.5 GB
- **Format**: DICOM + PNG
- **Views**: CC (Cranio-Caudal), MLO (Medio-Lateral Oblique)

---

## Usage

### After Downloading

**Run data preparation:**
```bash
python phase1_data_preparation/prepare_data.py
```

**Expected output:**
```
============================================================
Phase 1: Data Preparation
============================================================

Data Augmentation Enabled: True
Train/Val/Test Split: 0.7/0.15/0.15

📂 Loading BreakHis dataset from data/BreaKHis_v1...
Total samples: 7909
Benign: 5429
Malignant: 2480
✓ BreakHis: 7909 images loaded

============================================================
✓ Data Preparation Summary
============================================================

BreakHis:
  Total images: 7909
  Train: 5536 | Val: 1181 | Test: 1192
```

---

## Troubleshooting

### Download Too Slow?

**Option 1: Use wget (faster)**
```bash
wget http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz -O data/BreaKHis_v1.tar.gz
tar -xzf data/BreaKHis_v1.tar.gz -C data/
```

**Option 2: Use aria2 (parallel downloads)**
```bash
aria2c -x 16 http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz -o data/BreaKHis_v1.tar.gz
tar -xzf data/BreaKHis_v1.tar.gz -C data/
```

### Download Fails?

1. Check internet connection
2. Try alternative URL: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/
3. Use VPN if region-blocked
4. Download manually and place in `data/` folder

### Extraction Fails?

```bash
# Verify tar.gz file
tar -tzf data/BreaKHis_v1.tar.gz | head

# Try different extraction
tar -xvf data/BreaKHis_v1.tar.gz -C data/
```

### Dataset Not Found?

Check structure:
```bash
# Verify BreakHis directory
ls -la data/BreaKHis_v1/

# Should show: benign/ malignant/

# Count images
find data/BreaKHis_v1 -name "*.png" | wc -l
# Should show: 7909
```

---

## In Google Colab

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Download in Colab
!python download_breakhis.py

# Run data preparation
!python phase1_data_preparation/prepare_data.py
```

---

## Storage Requirements

| Dataset | Size | Compressed |
|---------|------|-----------|
| BreakHis | ~2 GB | 1.2 GB |
| CBIS-DDSM | ~7 GB | 3.5 GB |
| Both | ~9 GB | 4.7 GB |

**Recommendation**: Use Google Drive (15GB free) for Colab

---

## Next Steps

After downloading:
1. ✅ Run `python phase1_data_preparation/prepare_data.py`
2. ✅ Open `notebooks/Phase1_DataPreparation_Colab.ipynb`
3. ✅ Explore data augmentations
4. ✅ Create train/val/test splits
5. ✅ Start Phase 2 model training

---

## References

**BreakHis Dataset Paper**: https://arxiv.org/abs/1506.01497
- Title: "Breast cancer histopathological image analysis"
- Authors: Spanhol et al.

**Official Database**: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/

---

Need help? Check [SETUP.md](SETUP.md) or [RESOURCES.md](RESOURCES.md)
