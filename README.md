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
    Ensure the dataset is organized as follows:
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
* **ResNet-101**: Run `resnet101_train.py`.
* **ResNeXt-101**: Run `resnext101_train.py`.
    > *Note: Both scripts support automatic resuming from the last saved checkpoint in case of system interruptions.*

### 2. Ensemble Inference Phase
Ensure `best_model_resnet101.pth` and `best_resnext.pth` are in the working directory, then run:
```python
python ensemble_inference.py
