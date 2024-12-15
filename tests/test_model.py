import torch
import pytest
from model import CustomNet, DepthwiseSeparableConv

def test_model_output_shape():
    model = CustomNet()
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape (4, 10), got {output.shape}"

def test_model_parameters():
    model = CustomNet()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 200000, f"Model has {total_params} parameters, should be less than 200000"

def test_depthwise_separable_conv():
    in_channels, out_channels = 64, 128
    conv = DepthwiseSeparableConv(in_channels, out_channels)
    x = torch.randn(1, in_channels, 16, 16)
    output = conv(x)
    assert output.shape == (1, out_channels, 16, 16)
    
    # Test stride
    conv_stride = DepthwiseSeparableConv(in_channels, out_channels, stride=2)
    output_stride = conv_stride(x)
    assert output_stride.shape == (1, out_channels, 8, 8)

def test_dilated_convolution():
    model = CustomNet()
    # Check if conv2 has dilation
    assert model.conv2[0].dilation == (2, 2), "Conv2 should have dilation=2"

def test_model_forward_pass():
    model = CustomNet()
    x = torch.randn(1, 3, 32, 32)
    
    # Test each layer's output shape
    x1 = model.conv1(x)
    assert x1.shape == (1, 32, 32, 32), "Conv1 output shape incorrect"
    
    x2 = model.conv2(x1)
    assert x2.shape == (1, 64, 16, 16), "Conv2 output shape incorrect"
    
    x3 = model.conv3(x2)
    assert x3.shape == (1, 128, 8, 8), "Conv3 output shape incorrect"
    
    x4 = model.conv4(x3)
    assert x4.shape == (1, 256, 4, 4), "Conv4 output shape incorrect"
    
    x5 = model.gap(x4)
    assert x5.shape == (1, 256, 1, 1), "GAP output shape incorrect"

def calculate_rf_size(model):
    # Initial RF is 1x1
    rf_size = 1
    
    # Conv1: 3x3 kernel
    rf_size = rf_size + 2 * 1  # kernel_size=3, dilation=1
    
    # Conv2: 3x3 kernel with dilation=2
    rf_size = rf_size + 2 * 2 * 2  # kernel_size=3, dilation=2, stride=2 doubles effective RF
    
    # Conv3: 3x3 depthwise with stride=2
    rf_size = rf_size + 2 * 2  # kernel_size=3, stride=2 doubles effective RF
    
    # Conv4: 3x3 with stride=2
    rf_size = rf_size + 2 * 2  # kernel_size=3, stride=2 doubles effective RF
    
    return rf_size

def test_receptive_field():
    model = CustomNet()
    rf_size = calculate_rf_size(model)
    assert rf_size > 44, f"Receptive field size should be > 44, got {rf_size}" 