# CIFAR-10 Custom CNN Implementation

A PyTorch implementation of a custom CNN architecture for CIFAR-10 classification with specific architectural constraints and requirements.

## Table of Contents
1. [Architecture Features](#architecture-features)
2. [Data Augmentation](#data-augmentation)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training Features](#training-features)
7. [Test Coverage](#test-coverage)
8. [Requirements](#requirements)

## Architecture Features

### Network Design
* No MaxPooling (uses strided convolutions)
* Receptive Field > 44 pixels
* Uses Depthwise Separable Convolution
* Uses Dilated Convolution
* Global Average Pooling (GAP)
* Parameters < 200k

### Layer Details
1. **Conv1**: Regular 3x3 Convolution
   * Input: 32x32x3 → Output: 32x32x16
   * BatchNorm + ReLU
   * RF: 3x3

2. **Conv2**: Dilated 3x3 Convolution (stride=2)
   * Input: 32x32x16 → Output: 16x16x32
   * Dilation=2, BatchNorm + ReLU
   * RF: 7x7

3. **Conv3**: Depthwise Separable 3x3 Convolution (stride=2)
   * Input: 16x16x32 → Output: 8x8x64
   * BatchNorm + ReLU
   * RF: 15x15

4. **Conv4**: Regular 3x3 Convolution (stride=2)
   * Input: 8x8x64 → Output: 4x4x128
   * BatchNorm + ReLU
   * RF: 31x31

5. **GAP + FC**
   * Global Average Pooling: 4x4x128 → 1x1x128
   * Fully Connected: 128 → 10 classes

### Architecture Benefits
* **Efficiency**: Depthwise separable convolutions reduce parameters
* **Large RF**: Dilated convolutions increase receptive field efficiently
* **No Information Loss**: Strided convolutions instead of pooling
* **Regularization**: BatchNorm and ReLU after each convolution

## Data Augmentation
Using Albumentations library with:

1. **Horizontal Flip** (p=0.5)
   * Random horizontal flips for better generalization

2. **ShiftScaleRotate**
   * shift_limit=0.0625 (6.25% image size)
   * scale_limit=0.1 (±10% scaling)
   * rotate_limit=45° (±45 degrees rotation)
   * p=0.5 (50% probability)

3. **CoarseDropout** (Cutout implementation)
   * num_holes_range=(3, 6)
   * hole_height_range=(10, 20)
   * hole_width_range=(10, 20)
   * p=0.5 (50% probability)

## Project Structure

### Core Files
* `model.py`: CNN architecture implementation
  * CustomNet class with modular layer design
  * DepthwiseSeparableConv implementation
* `dataset.py`: CIFAR-10 dataset and augmentations
  * Custom dataset class with Albumentations
  * Proper normalization and transforms
* `utils.py`: Training utilities and metrics
  * Training and testing loops
  * Metrics plotting and logging
* `train.py`: Main training script
  * Training configuration
  * Model initialization
  * Learning rate scheduling

### Test Suite
Located in `tests/` directory:

1. **Model Tests** (`test_model.py`):
   * Architecture verification (no MaxPool, has Depthwise, has Dilated)
   * Parameter count check
   * Receptive field calculation
   * Output shape verification

2. **Dataset Tests** (`test_dataset.py`):
   * Dataset size verification
   * Data types and shapes
   * Augmentation parameters
   * Normalization values
   * Transform randomness

3. **Training Tests** (`test_training.py`):
   * Training functionality
   * Loss improvement
   * Model saving/loading

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```

The script will:
1. Train for up to 50 epochs
2. Save best model as 'best_model.pth'
3. Generate training plots
4. Stop early if 85% accuracy reached

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py
pytest tests/test_dataset.py
pytest tests/test_training.py

# Run with coverage report
pytest --cov=./ tests/
```

## Training Features

1. **Optimizer**: Adam
   * Initial LR: 0.001
   * ReduceLROnPlateau scheduling
   * Factor: 0.5
   * Patience: 5 epochs

2. **Training Monitoring**:
   * Progress bar with live metrics
   * Loss and accuracy plots
   * Best model saving
   * Early stopping at 85% accuracy


