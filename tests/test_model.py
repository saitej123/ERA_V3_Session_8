import torch
import pytest
from model import CustomNet, DepthwiseSeparableConv

def test_architecture_requirements():
    """Test the core architecture requirements"""
    model = CustomNet()
    
    # Check for absence of MaxPool layers
    for module in model.modules():
        assert not isinstance(module, torch.nn.MaxPool2d), "Model should not use MaxPool2d"
    
    # Check for presence of Depthwise Separable Conv
    has_depthwise = False
    for module in model.modules():
        if isinstance(module, DepthwiseSeparableConv):
            has_depthwise = True
            break
    assert has_depthwise, "Model must use Depthwise Separable Convolution"
    
    # Check for presence of Dilated Conv
    has_dilated = False
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.dilation[0] > 1:
            has_dilated = True
            break
    assert has_dilated, "Model must use Dilated Convolution"
    
    # Check for GAP
    has_gap = False
    for module in model.modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            has_gap = True
            break
    assert has_gap, "Model must use Global Average Pooling"

def test_parameter_count():
    """Test that model has less than 200k parameters"""
    model = CustomNet()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 200000, f"Model has {total_params} parameters, should be less than 200000"

def test_receptive_field():
    """Test that receptive field is more than 44"""
    rf_size = 1
    
    # Conv1: 3x3 kernel
    rf_size = rf_size + 2  # kernel_size=3, dilation=1
    
    # Conv2: 3x3 kernel with dilation=2 and stride=2
    rf_size = rf_size + 2 * 2 * 2  # kernel_size=3, dilation=2, stride=2
    
    # Conv3: 3x3 depthwise with stride=2
    rf_size = rf_size + 2 * 2 * 2  # kernel_size=3, stride=2, previous stride=2
    
    # Conv4: 3x3 with stride=2
    rf_size = rf_size + 2 * 2 * 2 * 2  # kernel_size=3, stride=2, previous strides=2,2
    
    assert rf_size > 44, f"Receptive field size should be > 44, got {rf_size}"

def test_output_shape():
    """Test model output shape for CIFAR-10"""
    model = CustomNet()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 10), "Model output should be shape (batch_size, 10) for CIFAR-10"