# HW1: Image Classification (100 Classes)

## Introduction
This project aims to solve a 100-class image classification task using deep residual architectures. To achieve high accuracy and surpass the **Strong Baseline (0.94)**, we implemented a dual-model ensemble strategy, leveraging the stability of **ResNet-101** and the enhanced feature extraction capabilities of **ResNeXt-101 (32x8d)**.

**Key Technical Highlights:**
* **Backbone Optimization**: Utilized ResNeXt, a variant of ResNet with **Grouped Convolutions**, to improve representational power.
* **Model Ensemble**: Applied Softmax probability averaging to combine predictions from multiple architectures.
* **Inference Strategy**: Implemented **Ten-Crop Test-Time Augmentation (TTA)** to ensure robust predictions across different image scales and positions.
* **Resource Management**: Developed a multi-stage training pipeline with manual checkpointing to handle Kaggle's hardware constraints (RAM/VRAM) and session timeouts.

## Environment Setup
* **Development Platform**: Kaggle Notebook (Python 3.12)
* **Accelerator**: NVIDIA T4 x2 (Dual GPU)
* **Required Libraries**:
    ```bash
    pip install torch torchvision pandas tqdm Pillow
    ```
* **Data Structure**:
    ```text
    /data/
    ├── train/
    ├── val/
    └── test/
    ```

## Usage
The workflow is divided into two phases: Training and Ensemble Inference.

### 1. Training Phase
Execute the training scripts for each backbone respectively. The scripts will automatically save the best model weights (`.pth`) and training logs (`.csv`).
* **ResNet-101**: Run `ResNet-101.py`.
* **ResNeXt-101**: Run `ResNeXt-101.py`.

### 2. Ensemble Inference Phase
Ensure `best_resnet.pth` and `best_resnext.pth` are in the working directory, then run:
```python
python Ensemble.py
```
This script applies Ten-Crop TTA and generates the final `prediction.csv` using a weighted average ($0.5:0.5$).

## Performance Snapshot

| Method | Best Val Acc | Public Score (Test) | Remarks |
| :--- | :---: | :---: | :--- |
| **ResNet-101 (Baseline)** | 0.8833 | 0.930 | Standard Residual Network |
| **ResNet-101 Variant (ResNeXt)** | 0.8667 | 0.941 | Modified with Grouped Convolutions |
| **Weighted Ensemble** | **N/A** | **0.950** | **Final Submission (0.5:0.5 Ratio)** |

## Training Hyperparameters

* **Optimizer**: AdamW (Weight Decay: $1 \times 10^{-2}$)
* **Learning Rate**: Initial $5 \times 10^{-5}$ with **CosineAnnealingLR** scheduler
* **Batch Size**: 64 (Effective size via Gradient Accumulation)
* **Data Augmentation**: TrivialAugmentWide + RandomErasing ($p=0.2$)
* **Regularization**: Label Smoothing (0.1) + Dropout (0.4)
* **Inference Technique**: **Ten-Crop TTA** (Test-Time Augmentation)
