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
##  Usage

### Training
### Evaluation
### Inference
##  Results
##  Citation
