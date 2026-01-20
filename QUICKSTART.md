# Quick Start Guide - Breast Cancer Identification Project

## 🚀 Getting Started (5 Minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/vsiva763-git/breast-cancer-identification.git
cd breast-cancer-identification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Google Colab Setup (Recommended)
1. Open: https://colab.research.google.com
2. Upload notebook: `notebooks/Phase1_DataPreparation_Colab.ipynb`
3. Run cells sequentially to:
   - Setup environment
   - Mount Google Drive
   - Prepare datasets
   - Create augmentations
   - Build DataLoaders

---

## 📚 5 Phases Overview

### Phase 1: Data Preparation ✅
**Status**: Ready to use
- **Notebook**: [Phase1_DataPreparation_Colab.ipynb](notebooks/Phase1_DataPreparation_Colab.ipynb)
- **Script**: [phase1_data_preparation/prepare_data.py](phase1_data_preparation/prepare_data.py)
- **What**: Load BreakHis + CBIS-DDSM, augment, split data
- **Output**: Train/Val/Test splits, augmented samples

### Phase 2: Model Development (In Progress)
**Next**: Start training lightweight models
- **Script**: [phase2_model_development/train.py](phase2_model_development/train.py)
- **Models**: EfficientNet-B0 + MobileNet-V3
- **Features**: Mixed precision, early stopping, checkpointing

### Phase 3: Multi-Modal Fusion (In Progress)
**Next**: Combine predictions from both modalities
- **Script**: [phase3_multimodal_fusion/fusion.py](phase3_multimodal_fusion/fusion.py)
- **Strategies**: Early fusion, late fusion, attention-based
- **Benefit**: 2-5% accuracy improvement

### Phase 4: Explainability (In Progress)
**Next**: Generate interpretability visualizations
- **Script**: [phase4_explainability/xai.py](phase4_explainability/xai.py)
- **Methods**: Grad-CAM, SHAP, attention weights
- **Output**: Heatmaps, feature importance

### Phase 5: Deployment (In Progress)
**Next**: Build production application
- **Frontend**: [Streamlit app](phase5_deployment/streamlit_app.py)
- **Backend**: [FastAPI API](phase5_deployment/api.py)
- **Deploy**: Streamlit Cloud (free!)

---

## 📊 Project Structure

```
breast-cancer-identification/
├── 📁 phase1_data_preparation/      # ✅ Ready
│   └── prepare_data.py
├── 📁 phase2_model_development/     # ⏳ In Progress
│   └── train.py
├── 📁 phase3_multimodal_fusion/     # ⏳ Coming
│   └── fusion.py
├── 📁 phase4_explainability/        # ⏳ Coming
│   └── xai.py
├── 📁 phase5_deployment/            # ⏳ Coming
│   ├── streamlit_app.py
│   └── api.py
├── 📁 utils/                        # ✅ Ready
│   ├── data_utils.py
│   └── model_utils.py
├── 📁 configs/                      # ✅ Ready
│   └── config.yaml
├── 📁 notebooks/                    # ✅ Ready
│   └── Phase1_DataPreparation_Colab.ipynb
├── requirements.txt                 # ✅ Ready
├── Dockerfile                       # ✅ Ready
├── .gitignore                       # ✅ Ready
└── README.md                        # ✅ Ready
```

---

## 🎯 Next Steps

### Immediate (Today)
1. ✅ Clone repository
2. ✅ Review project structure
3. ⏳ Run Phase 1 notebook in Colab

### Short Term (This Week)
1. ⏳ Download datasets (BreakHis, CBIS-DDSM)
2. ⏳ Prepare data splits
3. ⏳ Start Phase 2 model training

### Medium Term (Next 2-3 Weeks)
1. ⏳ Complete model training
2. ⏳ Implement multi-modal fusion
3. ⏳ Add explainability features

### Long Term (By End of Project)
1. ⏳ Build Streamlit dashboard
2. ⏳ Deploy to Streamlit Cloud
3. ⏳ Write documentation
4. ⏳ Submit project

---

## 🔗 Important Links

### Dataset Sources
- **BreakHis**: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/
- **CBIS-DDSM**: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-dataset
- **INbreast**: https://www.kaggle.com/datasets/raghavchaubey/inbreast

### Free Resources
- **Google Colab**: https://colab.research.google.com (Free GPU!)
- **Streamlit Cloud**: https://streamlit.io/cloud (Free hosting!)
- **GitHub**: https://github.com (Free repos!)

### IEEE Papers
- **EfficientNet**: https://arxiv.org/abs/1905.11946
- **MobileNetV3**: https://arxiv.org/abs/1905.02175
- **Grad-CAM**: https://arxiv.org/abs/1610.02055
- **Breast Cancer Detection**: https://arxiv.org/search/?query=breast+cancer+detection

---

## 💡 Key Concepts

### Multi-Modal Learning
Combining two different imaging types:
- **Histopathology**: Microscopic tissue analysis (BreakHis)
- **Mammography**: X-ray based imaging (CBIS-DDSM)

### Lightweight Models
Efficient architectures for edge deployment:
- **EfficientNet-B0**: 5.3M parameters
- **MobileNet-V3**: 5.4M parameters

### Attention-Based Fusion
Learning optimal weighting of each modality:
```
Weight = softmax([w_histo, w_mammo])
Output = w_histo * features_histo + w_mammo * features_mammo
```

### Explainability
Making AI decisions interpretable:
- **Grad-CAM**: Shows which regions the model focuses on
- **SHAP**: Explains feature importance
- **Attention Weights**: Shows modality contribution

---

## ⚙️ Configuration

Edit [configs/config.yaml](configs/config.yaml) to customize:

```yaml
data:
  augmentation:
    enabled: true
  splits:
    train: 0.7
    val: 0.15
    test: 0.15

models:
  histopathology:
    backbone: efficientnet_b0
    input_size: 224
    
  mammography:
    backbone: mobilenet_v3_small
    input_size: 224
    
  training:
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    mixed_precision: true
```

---

## 🆘 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch_size in config.yaml (32 → 16)

### Issue: Dataset not found
**Solution**: Download from official sources and extract to `data/` folder

### Issue: Slow training
**Solution**: Enable mixed_precision in config.yaml

### Issue: Import errors
**Solution**: `pip install -r requirements.txt --upgrade`

---

## 📝 Project Timeline

- **Week 1-2**: Data Preparation + Model Development (Phase 1-2)
- **Week 3-4**: Multi-Modal Fusion (Phase 3)
- **Week 5-6**: Explainability + Optimization (Phase 4)
- **Week 7-8**: Deployment + Documentation (Phase 5)
- **Week 9-10**: Testing + Final Submission

---

## ✨ Expected Results

| Metric | Target |
|--------|--------|
| Accuracy | 95-97% |
| Sensitivity | 94-96% |
| Specificity | 95-97% |
| F1-Score | 0.95-0.97 |
| Inference Time | 200-500ms |
| Model Size | 15-20MB |

---

## 📞 Support Resources

1. **Read Documentation**: Check README.md in each phase folder
2. **Review Code**: All modules are well-commented
3. **Check Configs**: Adjust settings in configs/config.yaml
4. **Test Locally**: Run scripts before deploying to Colab

---

## 🎉 You're All Set!

Start with Phase 1 notebook in Google Colab. Questions? Check the comprehensive README or review the code comments.

**Good Luck! 🚀**
