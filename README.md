# FishFaceID

This repository contains the code implementation for the FishFaceID framework, a comprehensive deep learning-based solution for non-invasive individual fish identification in aquaculture settings.

Dataset link: https://drive.google.com/drive/folders/1WvXhaOPScXOLUdZRAmx70dFbyJOu2B_N?usp=drive_link


## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Framework Architecture](#framework-architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)


##  Overview

FishFaceID is a modular deep learning framework designed for robust, scalable, and accurate individual fish identification in aquaculture environments. The system addresses key challenges in Precision Aquaculture (PA), including visual complexity, low-contrast underwater imaging, and diversity of aquatic species.


##  Key Features

- **Modular Architecture**: Configurable system that integrates various state-of-the-art deep learning models
- **Multi-Species Support**: Handles identification across diverse aquaculture species
- **Dual-View Recognition**: Supports both underwater and overhead camera perspectives
- **Novel Vim-FFID Model**: Tailored architecture with class-aware prompts and visual-prompt interaction

##  Framework Architecture

FishFaceID integrates three main components:

1. **Preprocessing Module**: Handles image preprocessing, data augmentation, and feature extraction
2. **Model Backend**: Flexible model selection including VMamba, Vision Mamba (Vim), and our novel Vim-FFID
3. **Post-processing Module**: Implements entropy-based re-ranking and decision making

##  Dataset

The FishFaceID benchmark dataset consists of:

- **Four representative species**:
  - Sea cucumber (*Apostichopus japonicus*)
  - Leopard coral grouper (*Plectropomus leopardus*)
  - Speckled blue grouper (*Epinephelus cyanopodus*)
  - Grass carp (*Ctenopharyngodon idella*)

- **Dual viewing angles**:
  - Underwater view
  - Overhead view

- **Dataset characteristics**:
  - High-resolution images
  - Multiple sessions for temporal robustness
  - Varied environmental conditions
  - Comprehensive annotation
Dataset link: https://drive.google.com/drive/folders/1WvXhaOPScXOLUdZRAmx70dFbyJOu2B_N?usp=drive_link


##  Models

### Vim-FFID Model

Our novel Vim-FFID architecture is specifically designed for aquaculture individual identification with the following key components:

1. **Class-Aware Prompt Learning**: 
   - Dynamic prompt creation for each class
   - Multiple prompts per class to capture intra-class variations

2. **Multi-Stage Visual-Prompt Interaction**:
   - Bi-directional feature interaction between image and prompt features
   - Hierarchical feature exchange for enhanced representation

3. **Entropy-Based Re-ranking**:
   - Uncertainty-aware decision making
   - Sub-prototype clustering for handling within-class variations

### Supported Models

- **Vim-FFID**: Our proposed model 
- **VMamba**
- **Vim**
- **Swin Transformer**
- **DeiT**

##  Installation

```bash
# Clone the repository
git clone https://github.com/zhenglab/FishFaceID
cd FishFaceID

# Create and activate conda environment
conda create -n fishfaceid python=3.8
conda activate fishfaceid

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for Mamba models
pip install mamba-ssm>=1.0.1
pip install triton>=2.0.0
```
## Usage

Before running the scripts, please make sure to:

1. Download the **FishFaceID** dataset.
2. Set `--data_root` to the root directory of the dataset on your machine.

### Training
**Train Vim-FFID on the Sea Cucumber – Overhead subset (example):**

```bash
python tools/train_vim_ffid.py \
  --config configs/sea_cucumber_overhead.yaml \
  --data_root /path/to/FishFaceID \
  --output_dir outputs/sea_cucumber_overhead
```

--config: path to the training configuration file.
Here we use the config tailored for the Sea Cucumber – Overhead subset.
--data_root: root directory of the FishFaceID dataset.
Replace /path/to/FishFaceID with the actual dataset path on your machine.
--output_dir: directory where logs, checkpoints, and TensorBoard files will be saved.
This command trains Vim-FFID on the Sea Cucumber – Overhead subset and saves the best-performing checkpoint (e.g. best_acc1.pth) under outputs/sea_cucumber_overhead/.
  
### Evaluation 
**Evaluate a trained model：**

```bash
python tools/eval_vim_ffid.py \
  --config configs/sea_cucumber_overhead.yaml \
  --checkpoint outputs/sea_cucumber_overhead/best_acc1.pth
```

--config: the same configuration file used during training, to keep data splits and model settings consistent.
--checkpoint: path to the trained model checkpoint to be evaluated.
This script loads the specified checkpoint and reports identification performance (e.g. Acc@1, Acc@5) on the evaluation split of the Sea Cucumber – Overhead subset.

##  Citation
If you find this repository or the FishFaceID dataset useful in your research, please cite:

Qinyue Zhang, Zhensheng Shi, Naizhe Sun, Yangfan Wang, Lingling Zhang,
Bo Wang, Xiaogang Xun, Bing Zheng, Haiyong Zheng,
FishFaceID: A deep learning-based non-invasive system for aquaculture organism individual identification,
Aquaculture, Volume 613, 743375, 2026.
https://doi.org/10.1016/j.aquaculture.2025.743375

```bash
@article{ZHANG2026743375,
title = {FishFaceID: A deep learning-based non-invasive system for aquaculture organism individual identification},
journal = {Aquaculture},
volume = {613},
pages = {743375},
year = {2026},
issn = {0044-8486},
doi = {https://doi.org/10.1016/j.aquaculture.2025.743375},
url = {https://www.sciencedirect.com/science/article/pii/S004484862501261X},
}
```
