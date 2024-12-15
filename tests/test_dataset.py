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
            print(f"max_holes: {transform.max_holes}")
            print(f"max_height: {transform.max_height}")
            print(f"max_width: {transform.max_width}")
            print(f"min_holes: {transform.min_holes}")
            print(f"min_height: {transform.min_height}")
            print(f"min_width: {transform.min_width}")
            print(f"fill_value: {transform.fill_value}")
            print(f"mask_fill_value: {transform.mask_fill_value}")

def test_augmentations():
    dataset = CIFAR10Dataset(train=True)
    torch.manual_seed(42)  # Set seed for reproducibility
    image1, _ = dataset[0]
    torch.manual_seed(43)  # Different seed
    image2, _ = dataset[0]  # Get same image again
    
    # Test that augmentations are random (images should be different)
    assert not torch.allclose(image1, image2, rtol=1e-3), "Augmentations should be random"

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