# CIFAR-10 Custom CNN Implementation

A PyTorch implementation of a custom CNN architecture for CIFAR-10 classification with specific architectural constraints and requirements.

## Table of Contents
- [Architecture Features](#architecture-features)
- [Data Augmentation](#data-augmentation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Training Features](#training-features)
- [Requirements](#requirements)
- [Test Coverage](#test-coverage)
- [License](#license)

## Architecture Features

### Network Design
- No MaxPooling (uses strided convolutions)
- Receptive Field > 44 pixels
- Uses Depthwise Separable Convolution
- Uses Dilated Convolution
- Global Average Pooling (GAP)
- Parameters < 200k

### Layer Details
1. **Conv1**: Regular 3x3 Convolution
   - Input: 32x32x3 → Output: 32x32x32
   - BatchNorm + ReLU
   - RF: 3x3

2. **Conv2**: Dilated 3x3 Convolution (stride=2)
   - Input: 32x32x32 → Output: 16x16x64
   - Dilation=2, BatchNorm + ReLU
   - RF: 7x7

3. **Conv3**: Depthwise Separable 3x3 Convolution (stride=2)
   - Input: 16x16x64 → Output: 8x8x128
   - BatchNorm + ReLU
   - RF: 15x15

4. **Conv4**: Regular 3x3 Convolution (stride=2)
   - Input: 8x8x128 → Output: 4x4x256
   - BatchNorm + ReLU
   - RF: 31x31

5. **GAP + FC**
   - Global Average Pooling: 4x4x256 → 1x1x256
   - Fully Connected: 256 → 10 classes
   - Final RF: >44x44

### Architecture Benefits
- **Efficiency**: Depthwise separable convolutions reduce parameters
- **Large RF**: Dilated convolutions increase receptive field efficiently
- **No Information Loss**: Strided convolutions instead of pooling
- **Regularization**: BatchNorm and ReLU after each convolution

## Data Augmentation
Using Albumentations library with:
- Horizontal Flip (p=0.5)
  - Random horizontal flips for better generalization
- ShiftScaleRotate
  - shift_limit=0.1 (10% image size)
  - scale_limit=0.1 (±10% scaling)
  - rotate_limit=15° (±15 degrees rotation)
  - p=0.5 (50% probability)
- CoarseDropout (Cutout implementation)
  - max_holes=1
  - max_height=16px
  - max_width=16px
  - min_holes=1
  - min_height=16px
  - min_width=16px
  - fill_value=dataset_mean (CIFAR-10 mean values)

## Project Structure

### Core Files
- `model.py`: CNN architecture implementation
  - CustomNet class with modular layer design
  - DepthwiseSeparableConv implementation
- `dataset.py`: CIFAR-10 dataset and augmentations
  - Custom dataset class with Albumentations
  - Proper normalization and transforms
- `utils.py`: Training utilities and metrics
  - Training and testing loops
  - Metrics plotting and logging
- `train.py`: Main training script
  - Training configuration
  - Model initialization
  - Learning rate scheduling

### Test Suite
Located in `tests/` directory:

1. **Model Tests** (`test_model.py`):
   - Output shape verification
   - Parameter count check (< 200k)
   - Depthwise separable convolution
   - Dilated convolution
   - Receptive field calculation (> 44px)
   - Layer-wise forward pass

2. **Dataset Tests** (`test_dataset.py`):
   - Dataset size verification (50k train, 10k test)
   - Data types and shapes (3x32x32 images)
   - Augmentation parameters
   - Normalization values (CIFAR-10 mean/std)
   - Transform randomness

3. **Training Tests** (`test_training.py`):
   - Training step functionality
   - Testing step functionality
   - Learning rate scheduling
   - Model saving/loading
   - Loss improvement verification

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

## CI/CD Pipeline

GitHub Actions workflow includes:
1. Code linting (flake8)
   - PEP 8 compliance
   - Code quality checks
2. Test execution
   - All test suites
   - Multiple Python versions
3. Coverage reporting
   - Minimum 80% coverage required
4. Multiple Python versions (3.8, 3.9)

## Training Features

1. **Optimizer**: Adam
   - Initial LR: 0.001
   - ReduceLROnPlateau scheduling
   - Factor: 0.5
   - Patience: 5 epochs

2. **Training Monitoring**:
   - Progress bar with live metrics
   - Loss and accuracy plots
   - Best model saving
   - Early stopping at 85% accuracy
