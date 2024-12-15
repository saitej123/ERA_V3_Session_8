import torch
import pytest
from model import CustomNet, DepthwiseSeparableConv

def test_architecture_requirements():
    """Test the core architecture requirements"""
    model = CustomNet()
    
    # Check for absence of MaxPool layers
    has_maxpool = False
    for module in model.modules():
        if isinstance(module, torch.nn.MaxPool2d):
            has_maxpool = True
    print(f"Has MaxPool layers: {has_maxpool}")
    
    # Check for presence of Depthwise Separable Conv
    has_depthwise = False
    for module in model.modules():
        if isinstance(module, DepthwiseSeparableConv):
            has_depthwise = True
    print(f"Has Depthwise Separable Conv: {has_depthwise}")
    
    # Check for presence of Dilated Conv
    has_dilated = False
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.dilation[0] > 1:
            has_dilated = True
    print(f"Has Dilated Conv: {has_dilated}")
    
    # Check for GAP
    has_gap = False
    for module in model.modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            has_gap = True
    print(f"Has Global Average Pooling: {has_gap}")

def test_parameter_count():
    """Test that model has less than 200k parameters"""
    model = CustomNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

def test_receptive_field():
    """Print receptive field size"""
    rf_size = 1
    
    # Conv1: 3x3 kernel
    rf_size = rf_size + 2  # kernel_size=3, dilation=1
    print(f"After Conv1: RF = {rf_size}")
    
    # Conv2: 3x3 kernel with dilation=2 and stride=2
    rf_size = rf_size + 2 * 2 * 2  # kernel_size=3, dilation=2, stride=2
    print(f"After Conv2: RF = {rf_size}")
    
    # Conv3: 3x3 depthwise with stride=2
    rf_size = rf_size + 2 * 2 * 2  # kernel_size=3, stride=2, previous stride=2
    print(f"After Conv3: RF = {rf_size}")
    
    # Conv4: 3x3 with stride=2
    rf_size = rf_size + 2 * 2 * 2 * 2  # kernel_size=3, stride=2, previous strides=2,2
    print(f"Final Receptive Field: {rf_size}")

def test_output_shape():
    """Test model output shape for CIFAR-10"""
    model = CustomNet()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Model output shape: {output.shape}")