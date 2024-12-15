# CIFAR-10 Custom Model

A PyTorch implementation of a custom CNN for CIFAR-10 classification.

## Test Cases

The model satisfies these conditions:

### 1. Architecture

- No max pooling layers
- Uses stride convolutions (3x3)
- Uses depth-wise separable convolution
- Uses dilated convolution
- Uses global average pooling
- Parameters < 200k

### 2. Receptive Field Analysis

The model achieves RF > 44 through:

```text
Layer1 (stride 2): RF = 3
Layer2 (stride 2): RF = 7
Layer3 (dilated 2): RF = 15
Layer4 (stride 2): RF = 31
Final RF = 47
```

### 3. Data Augmentation

Using `albumentations` with:

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    A.CoarseDropout(
        max_holes=1,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=16,
        min_width=16,
        fill_value=tuple([x * 255.0 for x in CIFAR_MEAN]),
        p=0.5
    ),
    A.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ToTensorV2()
])
```

### 4. Performance Metrics

- Target accuracy: > 85% on CIFAR-10 test set

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py
```

## Model Architecture

```text
Input (3, 32, 32)
├── Conv1: 3x3 s2 (3 → 16)
├── Depth-wise Conv: 3x3 s2 (16 → 32)
├── Dilated Conv: 3x3 d2 (32 → 64)
├── Conv4: 3x3 s2 (64 → 128)
├── Global Avg Pool
└── FC (128 → 10)
```

## Dependencies

```text
torch==2.5.1
torchvision==0.20.1
albumentations==1.4.22
numpy==1.26.4
loguru==0.7.2
opencv-python==4.10.0.84
tqdm==4.66.5
```

## Training Logs

The `training.log` file contains:

- Training/test loss and accuracy
- Test condition verification
- Parameter count
- Best model checkpoints


