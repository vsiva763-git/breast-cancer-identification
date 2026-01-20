# 📦 Kaggle Dataset Setup - BreakHis in Repository

## Overview

The Kaggle Training notebook now expects the BreakHis dataset to be **included in the GitHub repository** instead of downloading it separately. This simplifies the Kaggle workflow.

## Current Status

⚠️ **The dataset (8GB) is currently IGNORED by Git**

The `data/` directory is listed in `.gitignore`, so it won't be pushed to GitHub.

---

## ✅ Solution Options

### Option 1: Use Kaggle Dataset (Recommended)

Instead of putting 8GB in the GitHub repo, upload the dataset to Kaggle:

1. **Create a Kaggle Dataset:**
   ```bash
   cd /workspaces/breast-cancer-identification
   
   # Create dataset metadata
   cat > dataset-metadata.json <<EOF
   {
     "title": "BreakHis Breast Cancer Histopathology",
     "id": "yourusername/breakhis-breast-cancer",
     "licenses": [{"name": "other"}]
   }
   EOF
   
   # Initialize Kaggle API
   kaggle datasets init -p data/
   
   # Upload dataset (requires Kaggle API token)
   kaggle datasets create -p data/
   ```

2. **Update notebook to use Kaggle Dataset:**
   - Add input dataset in Kaggle notebook settings
   - Dataset will be available at `/kaggle/input/breakhis-breast-cancer/`
   - Update paths in notebook accordingly

### Option 2: Keep Dataset in Git (Not Recommended - 8GB!)

If you really want to commit the dataset to GitHub:

1. **Remove data/ from .gitignore:**
   ```bash
   # Edit .gitignore and remove these lines:
   # data/
   # datasets/
   ```

2. **Commit dataset:**
   ```bash
   git add data/BreaKHis_v1/
   git commit -m "Add BreakHis dataset"
   git push origin main
   ```

   ⚠️ **Warning:** 
   - GitHub has 100MB file size limit
   - Repo size limits apply
   - Very slow clone/push operations
   - Not recommended for 8GB datasets

### Option 3: Git LFS (Large File Storage)

Use Git LFS for large files:

1. **Install and setup Git LFS:**
   ```bash
   git lfs install
   git lfs track "data/**/*.png"
   git add .gitattributes
   ```

2. **Commit with LFS:**
   ```bash
   git add data/
   git commit -m "Add BreakHis dataset via LFS"
   git push origin main
   ```

   **Note:** GitHub LFS has bandwidth limits (1GB/month free)

---

## 🎯 Recommended Approach

**Use Kaggle Datasets** - upload once, reuse in all notebooks:

### Step-by-Step:

1. **Get Kaggle API credentials:**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Setup Kaggle API locally:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   
   # Install Kaggle package
   pip install kaggle
   ```

3. **Create and upload dataset:**
   ```bash
   cd /workspaces/breast-cancer-identification
   
   # Create metadata
   cat > data/dataset-metadata.json <<EOF
   {
     "title": "BreakHis Breast Cancer Histopathology Dataset",
     "id": "vsiva763/breakhis-breast-cancer",
     "subtitle": "Histopathological images for breast cancer classification",
     "description": "The BreakHis dataset contains 7,909 histopathology images of breast tumor tissue collected from 82 patients. Images are labeled as benign or malignant.",
     "isPrivate": false,
     "licenses": [
       {
         "name": "other"
       }
     ],
     "keywords": [
       "cancer",
       "health",
       "medicine",
       "image classification",
       "computer vision"
     ]
   }
   EOF
   
   # Upload (this may take 30-60 minutes for 8GB)
   kaggle datasets create -p data/ -r zip
   ```

4. **Update Kaggle notebook:**

   Replace the repository clone section with:
   
   ```python
   # Dataset is auto-mounted at /kaggle/input/
   import os
   from pathlib import Path
   
   # Check for Kaggle dataset
   dataset_path = Path('/kaggle/input/breakhis-breast-cancer')
   
   if dataset_path.exists():
       print("✓ Dataset found in Kaggle inputs!")
       # Create symlink to expected location
       !ln -s /kaggle/input/breakhis-breast-cancer data
   else:
       print("❌ Please add the dataset to this notebook:")
       print("   1. Click 'Add data' (right sidebar)")
       print("   2. Search: 'breakhis-breast-cancer'")
       print("   3. Click 'Add'")
   ```

5. **Add dataset to your Kaggle notebook:**
   - Open notebook on Kaggle
   - Click "Add data" in right sidebar
   - Search for your dataset name
   - Click "Add"

---

## 📋 Quick Decision Guide

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Kaggle Dataset** | • Separate from code<br>• Reusable across notebooks<br>• No GitHub limits | • Requires upload step<br>• Separate management | ✅ **Recommended** |
| **Git LFS** | • Version controlled<br>• Integrated with repo | • Bandwidth limits<br>• Slower operations | Large teams |
| **Direct Git** | • Simple workflow<br>• Everything in one place | • ❌ 8GB too large<br>• Slow operations<br>• GitHub limits | ❌ Not feasible |
| **Download Script** | • On-demand loading<br>• Smaller repo | • Download time<br>• External dependency | Quick testing |

---

## 🔄 Current Implementation Status

**Notebook Updated:** ✅
- Removed dataset download step
- Expects data in `data/BreaKHis_v1/`
- Added dataset verification

**Next Steps:**
1. Choose one of the options above
2. If using Kaggle Dataset: Create and upload dataset
3. If using Git: Update `.gitignore` and commit
4. Test on Kaggle to verify it works

---

## 📖 Additional Resources

- [Kaggle Datasets Documentation](https://www.kaggle.com/docs/datasets)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [BreakHis Original Source](http://www.inf.ufpr.br/vri/databases/)

---

**Questions?** Check the [main Kaggle guide](KAGGLE_GUIDE.md) for training instructions.
