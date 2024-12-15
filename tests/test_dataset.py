import torch
import pytest
import numpy as np
from dataset import CIFAR10Dataset, get_dataloaders

def test_dataset_length():
    """Print dataset lengths"""
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

def test_dataset_output():
    """Print dataset output information"""
    dataset = CIFAR10Dataset(train=True)
    image, label = dataset[0]
    
    print(f"Image type: {type(image)}")
    print(f"Image shape: {image.shape}")
    print(f"Label type: {type(label)}")
    print(f"Label value: {label}")

def test_dataloaders():
    """Print dataloader information"""
    train_loader, test_loader = get_dataloaders(batch_size=128)
    
    # Get first batch
    images, labels = next(iter(train_loader))
    print(f"Batch size: {images.shape[0]}")
    print(f"Image dimensions: {images.shape[1:]}")
    print(f"Image dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")

def test_augmentation_requirements():
    """Print augmentation parameters"""
    dataset = CIFAR10Dataset(train=True)
    
    # Get list of transform names
    transform_names = [type(t).__name__ for t in dataset.transform.transforms]
    print(f"Available transforms: {transform_names}")
    
    # Print CoarseDropout parameters
    for transform in dataset.transform.transforms:
        if type(transform).__name__ == 'CoarseDropout':
            print("\nCoarseDropout parameters:")
            print(f"n_holes: {transform.n_holes}")
            print(f"hole_height_range: {transform.hole_height_range}")
            print(f"hole_width_range: {transform.hole_width_range}")
            print(f"fill_value: {transform.fill_value}")
            print(f"mask_fill_value: {transform.mask_fill_value}")

def test_normalization():
    """Print normalization statistics"""
    dataset = CIFAR10Dataset(train=False)
    image, _ = dataset[0]
    
    print(f"Image mean: {image.mean():.4f}")
    print(f"Image std: {image.std():.4f}")

def test_cifar10_stats():
    """Print CIFAR-10 statistics"""
    dataset = CIFAR10Dataset(train=True)
    print(f"CIFAR-10 mean: {dataset.mean}")
    print(f"CIFAR-10 std: {dataset.std}")

def test_augmentation_randomness():
    """Print augmentation differences"""
    dataset = CIFAR10Dataset(train=True)
    torch.manual_seed(42)
    image1, _ = dataset[0]
    torch.manual_seed(43)
    image2, _ = dataset[0]
    
    diff = (image1 - image2).abs().mean().item()
    print(f"Average difference between augmented images: {diff:.4f}")