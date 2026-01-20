# 📚 Complete Resources & Links Guide

## 🎯 Project Overview
- **Repository**: https://github.com/vsiva763-git/breast-cancer-identification
- **Project Type**: End-Semester Project
- **Duration**: 8-10 weeks
- **Difficulty**: Intermediate-Advanced
- **Estimated Effort**: 80-120 hours

---

## 📊 Dataset Downloads

### BreakHis (Histopathology Images)
- **Official Source**: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/
- **Size**: ~1.2 GB
- **Images**: 7,909
- **Paper**: https://arxiv.org/abs/1506.01497
- **Classes**: 
  - Benign: Adenosis, Fibroadenoma, Phyllodes, Tubular
  - Malignant: Ductal, Lobular, Mucinous, Papillary
- **Magnifications**: 40×, 100×, 200×, 400×

### CBIS-DDSM (Mammography Images)
- **Kaggle Download**: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-dataset
- **Official Source**: https://www.cancer.gov/
- **Size**: ~3.5 GB
- **Images**: ~6,000
- **Views**: CC, MLO
- **Classes**: Normal, Benign, Malignant

### INbreast (Mammography)
- **Kaggle Download**: https://www.kaggle.com/datasets/raghavchaubey/inbreast
- **Size**: ~1 GB
- **Images**: 1,640
- **Classes**: Normal, Benign, Malignant

### Dataset Verification
- **MD5 Hash Checker**: https://www.md5file.com/
- **Data Validation**: Check provided checksums

---

## 🛠️ Development Tools & Platforms

### Cloud Computing (Free GPU)
- **Google Colab**: https://colab.research.google.com
  - Free 12GB GPU (Tesla T4 or P100)
  - Free TPU access
  - 12-hour runtime limit per session
  - Storage: Google Drive integration

- **Kaggle Notebooks**: https://www.kaggle.com/code
  - Free GPU access (40 hours/week)
  - Pre-loaded datasets
  - Community notebooks

- **AWS Free Tier**: https://aws.amazon.com/free/
  - 750 hours/month EC2 instance
  - SageMaker notebooks

- **Google Cloud Free Tier**: https://cloud.google.com/free
  - $300 free credits
  - Compute Engine, AI Platform

### Storage (Free)
- **Google Drive**: https://drive.google.com (15 GB free)
- **Dropbox**: https://www.dropbox.com (2 GB free)
- **OneDrive**: https://onedrive.live.com (5 GB free)

### Version Control
- **GitHub**: https://github.com (Free public repos)
- **GitLab**: https://gitlab.com (Free)
- **Gitea**: https://gitea.io (Self-hosted)

---

## 📚 Deep Learning Frameworks & Libraries

### Core Libraries
- **PyTorch**: https://pytorch.org (Deep learning framework)
- **PyTorch Lightning**: https://lightning.ai (Training framework)
- **TensorFlow**: https://tensorflow.org (Alternative)

### Model Zoo & Pre-trained Models
- **Timm (PyTorch Image Models)**: https://timm.fast.ai/
  - 1000+ pre-trained models
  - EfficientNet, MobileNet, ViT, etc.
  
- **HuggingFace Models**: https://huggingface.co/models
  - Vision models and datasets

### Data Augmentation
- **Albumentations**: https://albumentations.ai/
  - Fast GPU-accelerated augmentations
  - Medical imaging support
  
- **Torchvision**: https://pytorch.org/vision/stable/
  - Built-in transforms
  
- **MONAI**: https://monai.io/
  - Medical imaging toolkit

### Explainability (XAI)
- **Captum**: https://captum.ai/
  - Model interpretability
  - Grad-CAM, attention
  
- **SHAP**: https://shap.readthedocs.io/
  - Feature importance
  - LIME, TreeExplainer
  
- **OpenCV**: https://opencv.org/
  - Image processing
  - Heatmap visualization

### Hyperparameter Tuning
- **Optuna**: https://optuna.org/
  - Hyperparameter optimization
  - Pruning support
  
- **Ray Tune**: https://docs.ray.io/en/latest/tune/
  - Distributed tuning
  
- **Hyperopt**: http://hyperopt.github.io/hyperopt/
  - Bayesian optimization

### Experiment Tracking
- **Weights & Biases**: https://wandb.ai/
  - Free tier (500 GB)
  - ML experiment tracking
  
- **MLflow**: https://mlflow.org/
  - Open-source tracking
  - Self-hosted
  
- **Neptune.ai**: https://neptune.ai/
  - ML metadata platform

---

## 🎛️ Deployment Platforms

### Streamlit (Frontend)
- **Official Site**: https://streamlit.io/
- **Streamlit Cloud**: https://streamlit.io/cloud (Free hosting)
- **Documentation**: https://docs.streamlit.io/
- **Gallery**: https://streamlit.io/gallery

### FastAPI (Backend API)
- **Official Site**: https://fastapi.tiangolo.com/
- **Documentation**: https://fastapi.tiangolo.com/
- **Deployment Guide**: https://fastapi.tiangolo.com/deployment/

### Docker
- **Official Site**: https://docker.com/
- **Docker Hub**: https://hub.docker.com/
- **Documentation**: https://docs.docker.com/

### Cloud Hosting Options
- **Heroku**: https://www.heroku.com/
  - Free tier (limited)
  - Simple deployment
  
- **Replit**: https://replit.com/
  - Free hosting
  - Online IDE
  
- **Render**: https://render.com/
  - Free tier
  - Easy deployment
  
- **PythonAnywhere**: https://www.pythonanywhere.com/
  - Free tier for Python apps
  
- **AWS Lambda**: https://aws.amazon.com/lambda/
  - Serverless functions
  - Free tier included

---

## 📖 Academic Resources & Papers

### Key Reference Papers

**Architecture Papers:**
1. **EfficientNet**: https://arxiv.org/abs/1905.11946
   - Efficient CNN scaling
   - B0-B7 variants

2. **MobileNetV3**: https://arxiv.org/abs/1905.02175
   - Lightweight networks
   - Mobile deployment

3. **ResNet**: https://arxiv.org/abs/1512.03385
   - Residual networks
   - Deep learning foundation

**Explainability Papers:**
4. **Grad-CAM**: https://arxiv.org/abs/1610.02055
   - Visual explanations
   - Class activation mapping

5. **SHAP**: https://arxiv.org/abs/1705.07874
   - Explainable AI
   - Game theory approach

6. **LIME**: https://arxiv.org/abs/1602.04938
   - Local interpretable models

**Medical Imaging Papers:**
7. **BreakHis Dataset**: https://arxiv.org/abs/1506.01497
   - Histopathological images
   - Benchmark dataset

8. **U-Net**: https://arxiv.org/abs/1505.04597
   - Medical image segmentation

**Multi-Modal Learning:**
9. **Attention Mechanisms**: https://arxiv.org/abs/1706.03762
   - Transformer architecture
   - Attention fusion

### Paper Search Engines
- **arXiv.org**: https://arxiv.org/
  - Preprints repository
  - `https://arxiv.org/search/?query=breast+cancer+detection`
  
- **IEEE Xplore**: https://ieeexplore.ieee.org/
  - IEEE papers (some free)
  - `https://ieeexplore.ieee.org/Xplore/home.jsp`
  
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/
  - Medical literature
  - Free full texts (PubMed Central)
  
- **Google Scholar**: https://scholar.google.com/
  - Academic search
  - PDF links
  
- **Semantic Scholar**: https://www.semanticscholar.org/
  - AI-powered search
  - Citation tracking

### Conferences & Journals
- **MICCAI**: https://www.miccai.org/
  - Medical image computing
  
- **IEEE ISBI**: https://biomedicalimaging.org/
  - Biomedical imaging
  
- **NeurIPS**: https://nips.cc/
  - Machine learning conference
  
- **CVPR**: https://cvpr2024.thecvf.com/
  - Computer vision conference

---

## 🎓 Learning Resources

### Python & Deep Learning Tutorials
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Fast.ai**: https://www.fast.ai/
- **Kaggle Learn**: https://www.kaggle.com/learn
- **DataCamp**: https://www.datacamp.com/
- **Coursera ML**: https://www.coursera.org/

### Medical AI
- **MONAI Tutorials**: https://monai.io/tutorial.html
- **Stanford ML4Health**: https://stanfordmlgroup.github.io/
- **MIT Healthcare AI**: https://healthcareailab.mit.edu/

### Debugging & Profiling
- **PyTorch Profiler**: https://pytorch.org/docs/stable/profiler.html
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **Weights & Biases Profiling**: https://docs.wandb.ai/

---

## 🔧 Useful Tools & Utilities

### Development IDEs
- **VS Code**: https://code.visualstudio.com/ (Free, Recommended)
  - Python extension
  - Jupyter support
  - GitHub Copilot
  
- **PyCharm**: https://www.jetbrains.com/pycharm/
  - Professional IDE
  - Community edition (free)
  
- **Jupyter Lab**: https://jupyter.org/
  - Notebook interface
  - Lab environment
  
- **Google Colab**: https://colab.research.google.com/
  - Online notebook
  - Free GPU

### Package Managers
- **Pip**: https://pip.pypa.io/ (Python)
- **Conda**: https://conda.io/ (Package + environment)
- **Mamba**: https://mamba.readthedocs.io/ (Faster conda)

### Environment Management
- **venv**: Built-in Python
- **Conda**: https://conda.io/
- **Poetry**: https://python-poetry.org/
- **Pipenv**: https://pipenv.pypa.io/

### Code Quality
- **Black**: https://github.com/psf/black (Code formatting)
- **Flake8**: https://flake8.pycqa.org/ (Linting)
- **Mypy**: http://mypy-lang.org/ (Type checking)
- **Pre-commit**: https://pre-commit.com/

---

## 📱 Community & Forums

### Q&A Platforms
- **Stack Overflow**: https://stackoverflow.com/
- **Reddit r/MachineLearning**: https://www.reddit.com/r/MachineLearning/
- **Reddit r/learnprogramming**: https://www.reddit.com/r/learnprogramming/
- **Kaggle Discussion**: https://www.kaggle.com/discussion

### Slack Communities
- **PyTorch**: https://pytorch.org/slack/
- **Fast.ai**: https://forums.fast.ai/
- **Kaggle**: https://kaggle.com/

### Discord Communities
- **PyTorch**: https://discord.gg/pytorch
- **MONAI**: https://monai.io/

---

## 🚀 Deployment Guides

### Docker Deployment
- **Official Tutorial**: https://docs.docker.com/get-started/
- **Docker Hub**: https://hub.docker.com/
- **Dockerfile Reference**: https://docs.docker.com/engine/reference/builder/

### Kubernetes
- **Kubernetes Tutorial**: https://kubernetes.io/docs/tutorials/
- **Kubeflow**: https://www.kubeflow.org/ (ML on Kubernetes)

### Model Serving
- **BentoML**: https://bentoml.io/
- **Seldon Core**: https://www.seldon.io/
- **TorchServe**: https://pytorch.org/serve/

---

## 📊 Monitoring & Analytics

### Model Monitoring
- **Weights & Biases Reports**: https://docs.wandb.ai/
- **Prometheus**: https://prometheus.io/
- **Grafana**: https://grafana.com/

### Performance Analysis
- **PyTorch Profiler**: https://pytorch.org/docs/stable/profiler.html
- **NVIDIA Nsight**: https://developer.nvidia.com/nsight
- **Memory Profiler**: https://github.com/pythonprofilers/memory_profiler

---

## 💰 Cost Estimation

### Free Options (Recommended)
- Google Colab GPU: FREE
- Google Drive Storage: 15 GB FREE
- GitHub Repos: FREE
- Streamlit Cloud Hosting: FREE
- Datasets: FREE (public)

**Total Cost: $0**

### Paid Options (Optional)
- Google Cloud Credits: $300
- AWS Credits: $100-300
- Kaggle GPU (extra hours): $20/month
- Streamlit Pro: $40-100/month

---

## ✅ Pre-Launch Checklist

Before starting development:

- [ ] Have GitHub account
- [ ] Access Google Colab
- [ ] Have Google Drive access
- [ ] Download datasets (or know where to find them)
- [ ] Review PyTorch documentation
- [ ] Read project README.md
- [ ] Understand 5-phase architecture
- [ ] Check system requirements
- [ ] Install required packages locally
- [ ] Test Colab environment

---

## 📞 Quick Reference Links

**Documentation**:
- Project README: [README.md](README.md)
- Quick Start: [QUICKSTART.md](QUICKSTART.md)
- Setup Guide: [SETUP.md](SETUP.md)
- This Resources: [RESOURCES.md](RESOURCES.md)

**Development**:
- Phase 1 Notebook: [Phase1_DataPreparation_Colab.ipynb](notebooks/Phase1_DataPreparation_Colab.ipynb)
- Configuration: [config.yaml](configs/config.yaml)
- Requirements: [requirements.txt](requirements.txt)

**Repository**:
- GitHub: https://github.com/vsiva763-git/breast-cancer-identification
- Issues: https://github.com/vsiva763-git/breast-cancer-identification/issues
- Discussions: https://github.com/vsiva763-git/breast-cancer-identification/discussions

---

## 🎉 Ready to Start?

1. **Open Google Colab**: https://colab.research.google.com
2. **Upload Notebook**: Open Phase1_DataPreparation_Colab.ipynb
3. **Follow Along**: Execute cells step by step
4. **Experiment**: Modify and test
5. **Progress**: Move to Phase 2 when ready

---

**Happy Learning! 🚀**

*All resources are free and open-source. Save this page for quick reference!*
