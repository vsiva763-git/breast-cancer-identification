# Complete Setup & Installation Guide

## System Requirements

- **OS**: Linux, macOS, or Windows (WSL2)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended (NVIDIA with CUDA 11.8+)

## Installation Methods

### Method 1: Local Setup (Linux/macOS)

#### 1.1 Clone Repository
```bash
git clone https://github.com/vsiva763-git/breast-cancer-identification.git
cd breast-cancer-identification
```

#### 1.2 Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cancer-id python=3.10
conda activate cancer-id
```

#### 1.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 1.4 Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import timm; print(f'Timm: {timm.__version__}')"
python -c "import cuda=True if torch.cuda.is_available() else False; print(f'CUDA Available: {cuda}')"
```

---

### Method 2: Google Colab (Recommended for Free GPU)

#### 2.1 Open Colab
Visit: https://colab.research.google.com

#### 2.2 Create New Notebook
Click "New notebook"

#### 2.3 Clone Repository
```python
!git clone https://github.com/vsiva763-git/breast-cancer-identification.git
%cd breast-cancer-identification
```

#### 2.4 Install Dependencies
```python
!pip install -q -r requirements.txt
```

#### 2.5 Mount Google Drive (Optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 2.6 Verify GPU
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

### Method 3: Docker

#### 3.1 Build Image
```bash
docker build -t breast-cancer-id:latest .
```

#### 3.2 Run Container
```bash
# For interactive development
docker run -it --gpus all \
  -v $(pwd):/app \
  -p 8000:8000 \
  -p 8501:8501 \
  breast-cancer-id:latest

# For Streamlit app
docker run --gpus all \
  -p 8501:8501 \
  breast-cancer-id:latest \
  streamlit run phase5_deployment/streamlit_app.py
```

---

### Method 4: GitHub Codespaces

#### 4.1 Open Repository
Visit: https://github.com/vsiva763-git/breast-cancer-identification

#### 4.2 Create Codespace
Click "Code" → "Codespaces" → "Create codespace on main"

#### 4.3 Wait for Initialization
The environment will automatically set up

#### 4.4 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### BreakHis Dataset

#### Option A: Download Manually
1. Visit: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/
2. Download the dataset
3. Extract to: `data/BreakHis/`
4. Structure should be:
```
data/BreakHis/
├── benign/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── malignant/
    ├── image1.png
    ├── image2.png
    └── ...
```

#### Option B: Download via Script (Colab)
```python
import requests
import zipfile

# Download from source
url = "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/"
# Follow website instructions

# Extract
zipfile.ZipFile('BreakHis.zip').extractall('data/BreakHis')
```

### CBIS-DDSM Dataset

#### From Kaggle
1. Get Kaggle API key from: https://www.kaggle.com/settings/account
2. Install kaggle: `pip install kaggle`
3. Place API key: `~/.kaggle/kaggle.json`
4. Download:
```bash
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-dataset
unzip cbis-ddsm-breast-cancer-dataset.zip -d data/CBIS-DDSM/
```

### Verify Datasets
```bash
python -c "
from pathlib import Path
import os

datasets = {
    'BreakHis': 'data/BreakHis',
    'CBIS-DDSM': 'data/CBIS-DDSM'
}

for name, path in datasets.items():
    if Path(path).exists():
        count = len(list(Path(path).rglob('*.png'))) + len(list(Path(path).rglob('*.jpg')))
        print(f'✓ {name}: {count} images found')
    else:
        print(f'✗ {name}: Not found at {path}')
"
```

---

## Configuration

### Edit configs/config.yaml

```yaml
# Data paths
data:
  root_path: ./data
  
  datasets:
    - name: BreakHis
      path: data/BreakHis
      modality: histopathology
      
    - name: CBIS-DDSM
      path: data/CBIS-DDSM
      modality: mammography
  
  # Augmentation settings
  augmentation:
    enabled: true
    techniques:
      - horizontal_flip
      - vertical_flip
      - rotation: 30
      - gaussian_blur
  
  # Split ratio
  splits:
    train: 0.7
    val: 0.15
    test: 0.15
    balanced: true

# Model configuration
models:
  histopathology:
    backbone: efficientnet_b0
    pretrained: true
    input_size: 224
    dropout: 0.3
    
  mammography:
    backbone: mobilenet_v3_small
    pretrained: true
    input_size: 224
    dropout: 0.3

  # Training hyperparameters
  training:
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    optimizer: adam
    scheduler: cosine
    mixed_precision: true
    early_stopping: true
    patience: 5

# Fusion strategy
fusion:
  strategy: attention_based  # early, late, attention_based
  attention_mechanism: true

# Explainability
explainability:
  gradcam: true
  shap: true
  visualization: true

# Deployment
deployment:
  framework: streamlit
  api_framework: fastapi
  docker: true
  model_compression: true
  quantization: int8
```

---

## Verification Scripts

### Check Installation
```bash
python -c "
import sys
print('Python:', sys.version)

import torch
print('PyTorch:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())

import timm
print('Timm:', timm.__version__)

import pandas as pd
print('Pandas:', pd.__version__)

import albumentations as A
print('Albumentations:', A.__version__)

print('\n✓ All dependencies installed successfully!')
"
```

### Check GPU
```bash
python -c "
import torch

if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'CUDA Version: {torch.version.cuda}')
else:
    print('No GPU detected. Using CPU.')
"
```

### Check Datasets
```bash
python -c "
from pathlib import Path

datasets = ['data/BreakHis', 'data/CBIS-DDSM']
for ds in datasets:
    p = Path(ds)
    if p.exists():
        images = list(p.rglob('*.png')) + list(p.rglob('*.jpg'))
        print(f'✓ {ds}: {len(images)} images')
    else:
        print(f'✗ {ds}: Not found')
"
```

---

## Running the Project

### Phase 1: Data Preparation
```bash
# Run locally
python phase1_data_preparation/prepare_data.py

# Or in Colab
# Open: notebooks/Phase1_DataPreparation_Colab.ipynb
```

### Phase 2: Model Training
```bash
python phase2_model_development/train.py
```

### Phase 3: Multi-Modal Fusion
```bash
python phase3_multimodal_fusion/train_fusion.py
```

### Phase 4: Explainability
```bash
python phase4_explainability/generate_explanations.py
```

### Phase 5: Deployment

**Streamlit App**:
```bash
streamlit run phase5_deployment/streamlit_app.py
```

**FastAPI Server**:
```bash
python phase5_deployment/api.py
# Open: http://localhost:8000/docs
```

**Docker**:
```bash
docker build -t cancer-id .
docker run -p 8000:8000 -p 8501:8501 cancer-id
```

---

## Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Solution: Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

### Issue: CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
# In config.yaml: batch_size: 16

# Solution 2: Enable gradient checkpointing
# In train.py: model.gradient_checkpointing_enable()

# Solution 3: Use CPU
# Set device = 'cpu' in code
```

### Issue: Slow Training
```bash
# Enable mixed precision in config.yaml
mixed_precision: true

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Use more workers in DataLoader
DataLoader(dataset, num_workers=4)
```

### Issue: Dataset Files Not Found
```bash
# Check file structure
python -c "
from pathlib import Path
for p in Path('data').rglob('*'):
    print(p)
"
```

### Issue: Import Errors in Colab
```python
# Run in Colab cell
!pip install --upgrade setuptools wheel
!pip install -q -r requirements.txt --no-cache-dir
```

---

## Performance Optimization

### For Better Training Speed
1. Enable mixed precision: `mixed_precision: true`
2. Increase number of workers: `num_workers=4`
3. Use pinned memory: `pin_memory=True` in DataLoader
4. Enable gradient checkpointing: `model.gradient_checkpointing_enable()`

### For Better Accuracy
1. Increase epochs: `epochs: 100`
2. Use stronger augmentation: Add more techniques
3. Increase model capacity: Use larger backbone
4. Implement ensemble: Combine multiple models

### For Inference Speed
1. Use quantization: `quantization: int8`
2. Use pruning: Remove 30-50% of weights
3. Use TorchScript: Export to optimized format
4. Batch inference: Process multiple samples

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Download datasets
3. ✅ Run verification scripts
4. ⏳ Start Phase 1 notebook
5. ⏳ Complete project phases

---

## Support

- **Documentation**: Read comprehensive README.md
- **Issues**: Check GitHub issues
- **Discussion**: Review project forum
- **Email**: Submit issues on GitHub

---

**You're ready to start! 🚀**

Begin with Phase 1 in Google Colab or run locally.
