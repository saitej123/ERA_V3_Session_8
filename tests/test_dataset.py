import torch
import pytest
import numpy as np
from dataset import CIFAR10Dataset, get_dataloaders

def test_dataset_length():
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    assert len(train_dataset) == 50000, "Training dataset should have 50000 samples"
    assert len(test_dataset) == 10000, "Test dataset should have 10000 samples"

def test_dataset_output():
    dataset = CIFAR10Dataset(train=True)
    image, label = dataset[0]
    
    assert isinstance(image, torch.Tensor), "Dataset should return torch.Tensor for image"
    assert isinstance(label, int), "Dataset should return int for label"
    assert image.shape == (3, 32, 32), f"Image shape should be (3, 32, 32), got {image.shape}"
    assert 0 <= label <= 9, f"Label should be between 0 and 9, got {label}"

def test_dataloaders():
    train_loader, test_loader = get_dataloaders(batch_size=128)
    
    # Test batch size
    images, labels = next(iter(train_loader))
    assert images.shape[0] == 128, "Train loader batch size should be 128"
    assert labels.shape[0] == 128, "Train loader label batch size should be 128"
    
    # Test image dimensions
    assert images.shape[1:] == (3, 32, 32), "Image dimensions should be (3, 32, 32)"
    
    # Test data type
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.long, "Labels should be long"

def test_augmentation_parameters():
    dataset = CIFAR10Dataset(train=True)
    
    # Test CoarseDropout parameters
    coarse_dropout = None
    for transform in dataset.transform.transforms:
        if transform.__class__.__name__ == 'CoarseDropout':
            coarse_dropout = transform
            break
    
    assert coarse_dropout is not None, "CoarseDropout augmentation not found"
    assert coarse_dropout.max_holes == 1, "CoarseDropout max_holes should be 1"
    assert coarse_dropout.max_height == 16, "CoarseDropout max_height should be 16"
    assert coarse_dropout.max_width == 16, "CoarseDropout max_width should be 16"
    assert coarse_dropout.min_holes == 1, "CoarseDropout min_holes should be 1"
    assert coarse_dropout.min_height == 16, "CoarseDropout min_height should be 16"
    assert coarse_dropout.min_width == 16, "CoarseDropout min_width should be 16"

def test_augmentations():
    dataset = CIFAR10Dataset(train=True)
    image1, _ = dataset[0]
    image2, _ = dataset[0]  # Get same image again
    
    # Test that augmentations are random (images should be different)
    assert not torch.allclose(image1, image2), "Augmentations should be random"

def test_normalization():
    dataset = CIFAR10Dataset(train=False)  # Use test set to avoid augmentations
    image, _ = dataset[0]
    
    # Test if the image is normalized (mean close to 0, std close to 1)
    assert -0.5 <= image.mean() <= 0.5, "Image mean should be approximately 0"
    assert 0.5 <= image.std() <= 1.5, "Image std should be approximately 1"

def test_cifar10_mean_std():
    dataset = CIFAR10Dataset(train=True)
    assert np.allclose(dataset.mean, (0.4914, 0.4822, 0.4465), atol=1e-4), "Incorrect CIFAR-10 mean"
    assert np.allclose(dataset.std, (0.2470, 0.2435, 0.2616), atol=1e-4), "Incorrect CIFAR-10 std" 