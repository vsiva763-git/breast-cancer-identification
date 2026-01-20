# 🎓 Google Colab Training Guide

## Quick Start (5 minutes)

### 1️⃣ Open Notebook in Colab

Click this button to open the complete training notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vsiva763-git/breast-cancer-identification/blob/main/notebooks/Complete_Training_Colab.ipynb)

**Or manually:**
1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub tab
3. Enter: `vsiva763-git/breast-cancer-identification`
4. Select: `notebooks/Complete_Training_Colab.ipynb`

---

## 2️⃣ Enable GPU

**IMPORTANT**: You must enable GPU for training!

1. Click **Runtime** → **Change runtime type**
2. Hardware accelerator: Select **T4 GPU** (or P100 if available)
3. Click **Save**

Verify GPU:
```python
!nvidia-smi
```

---

## 3️⃣ Run All Cells

**Option A: Run All (Recommended)**
- Click **Runtime** → **Run all**
- Go get coffee ☕ (takes 2-3 hours)

**Option B: Run Step-by-Step**
- Run each cell sequentially (Shift+Enter)
- Good for understanding each step

---

## 📊 What Happens During Training

### Timeline (Total: ~2-3 hours)

| Step | Duration | What's Happening |
|------|----------|------------------|
| 1. Setup | 2 min | Install packages, mount Drive |
| 2. Download Dataset | 15-20 min | Download 1.2 GB BreakHis dataset |
| 3. Data Prep | 5 min | Load and split data |
| 4. Training | 2-3 hours | Train EfficientNet-B0 (50 epochs) |
| 5. Evaluation | 2 min | Test model, generate metrics |
| 6. Save Results | 1 min | Save to Google Drive |

### Expected Output

**After Training:**
```
Epoch 50/50
Train Loss: 0.1234  Train Acc: 96.5%
Val Loss: 0.1567    Val Acc: 95.8%

✓ Best model saved: checkpoints/best_model.pth
✓ Validation Accuracy: 95.8%
```

---

## 💡 Tips & Tricks

### 1. Keep Colab Active
Colab disconnects after ~90 minutes of inactivity.

**Solution**: Run this in a separate cell to keep alive:
```python
# Keep-alive script (run in background)
import time
while True:
    time.sleep(60)
```

### 2. Use Weights & Biases for Monitoring
Track experiments in real-time:
```python
import wandb
wandb.login()  # Get API key from wandb.ai
```

### 3. Quick Test Run First
Before full training, test with 5 epochs (~15 min):
```python
# In the notebook, this is done automatically
config['models']['training']['epochs'] = 5
```

### 4. Save Checkpoints Frequently
The notebook auto-saves to Drive after training, but you can manually save:
```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')
```

---

## 🎯 Training Options

### Option 1: Quick Test (Recommended First)
```python
# In notebook cell for "Quick Training"
!python phase2_model_development/train.py --config configs/config_test.yaml
```
- **Time**: 15-20 minutes
- **Epochs**: 5
- **Purpose**: Verify everything works

### Option 2: Full Training
```python
# In notebook cell for "Full Training"
!python phase2_model_development/train.py --config configs/config.yaml
```
- **Time**: 2-3 hours
- **Epochs**: 50
- **Expected Accuracy**: 95-97%

---

## 📁 Where Are My Results?

All results are automatically saved to:
```
/content/drive/MyDrive/breast_cancer_results_<timestamp>/
├── best_model.pth          # Trained model weights
├── config.yaml             # Configuration used
├── training.log            # Training logs
└── training_curves.png     # Accuracy/loss plots
```

---

## 🐛 Troubleshooting

### Problem: "Runtime disconnected"
**Cause**: Colab timeout or GPU limit reached
**Solution**: 
- Reconnect and re-run from checkpoint
- Use keep-alive script above

### Problem: "Out of memory"
**Cause**: Batch size too large
**Solution**: Reduce batch size in config:
```python
config['models']['training']['batch_size'] = 16  # Instead of 32
```

### Problem: "Dataset not found"
**Cause**: Download failed or incomplete
**Solution**: Re-run download cell:
```python
!python download_breakhis.py
```

### Problem: "Low accuracy (<90%)"
**Possible causes**:
- Not enough epochs (need 40-50)
- Learning rate too high/low
- Data augmentation too aggressive

**Solution**: Check training curves, adjust hyperparameters

---

## 🚀 After Training

### 1. Download Trained Model
```python
from google.colab import files
files.download('checkpoints/best_model.pth')
```

### 2. Test on New Images
Use the prediction cell in notebook to test on any image:
```python
prediction, confidence = predict_image('path/to/image.png')
print(f"{prediction}: {confidence:.2%}")
```

### 3. Deploy Model
- Copy `best_model.pth` to your local machine
- Run Streamlit app: `streamlit run phase5_deployment/streamlit_app.py`

---

## 📚 Next Steps

After successful training:

1. **Phase 3**: Multi-modal fusion (optional)
2. **Phase 4**: Explainability (Grad-CAM visualization)
3. **Phase 5**: Deploy as web app
4. **Documentation**: Write project report

---

## 💰 Colab Free vs Pro

| Feature | Free | Pro |
|---------|------|-----|
| GPU Access | T4 (limited) | A100/V100 |
| Session Length | 12 hours | 24 hours |
| RAM | 12 GB | 25 GB |
| Training Time | 2-3 hours | 1-1.5 hours |
| **Recommendation** | ✅ Sufficient | Optional |

**For this project, Colab Free is enough!**

---

## 🎓 For End-Semester Project

### What to Include in Report:

1. **Dataset**: BreakHis (7,909 images)
2. **Model**: EfficientNet-B0 with transfer learning
3. **Results**: 
   - Training accuracy: XX%
   - Validation accuracy: XX%
   - Test accuracy: XX%
4. **Training Details**:
   - 50 epochs, batch size 32
   - Mixed precision training
   - Early stopping with patience=5
5. **Visualizations**:
   - Training curves (from notebook)
   - Confusion matrix
   - Sample predictions

### Grading Criteria Checklist:
- ✅ Dataset properly prepared
- ✅ Model architecture documented
- ✅ Training process logged
- ✅ Results >95% accuracy
- ✅ Proper evaluation metrics
- ✅ Clear documentation

---

## 📞 Need Help?

- **Issues**: https://github.com/vsiva763-git/breast-cancer-identification/issues
- **Dataset**: http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz
- **Colab Docs**: https://colab.research.google.com/notebooks/intro.ipynb

---

## ⏱️ Time Estimates

| Task | Colab Free (T4) | Colab Pro (A100) |
|------|----------------|------------------|
| Dataset download | 15-20 min | 10-15 min |
| Data preparation | 5 min | 3 min |
| Training (50 epochs) | 2-3 hours | 1-1.5 hours |
| Evaluation | 2 min | 1 min |
| **Total** | **~3 hours** | **~1.5 hours** |

**Plan accordingly**: Start training in the morning, results by afternoon!

---

**Happy Training! 🎉**
