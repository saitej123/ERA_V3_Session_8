import torch
import pytest
import os
from model import CustomNet
from utils import train, test
import torch.nn as nn
import torch.optim as optim

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)
        self.targets = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def test_training_functions():
    """Test basic training functionality"""
    model = CustomNet()
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data
    train_dataset = DummyDataset(32)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    
    # Test training step
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch=1)
    
    assert isinstance(train_loss, float), "Training loss should be a float"
    assert isinstance(train_acc, float), "Training accuracy should be a float"
    assert 0 <= train_loss <= 10, "Training loss should be reasonable"
    assert 0 <= train_acc <= 100, "Training accuracy should be between 0 and 100"

def test_loss_improvement():
    """Test that loss decreases during training"""
    model = CustomNet()
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create small dataset for overfitting test
    train_dataset = DummyDataset(16)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
    
    # Initial loss
    initial_loss, _ = train(model, device, train_loader, criterion, optimizer, epoch=1)
    
    # Train for a few epochs
    for epoch in range(2, 4):
        final_loss, _ = train(model, device, train_loader, criterion, optimizer, epoch)
    
    assert final_loss < initial_loss, "Loss should decrease during training"

def test_model_saving(tmp_path):
    """Test model saving and loading"""
    model = CustomNet()
    save_path = f"{tmp_path}/model.pth"
    
    # Save model
    torch.save(model.state_dict(), save_path)
    assert os.path.exists(save_path), "Model file should exist after saving"
    
    # Load model
    loaded_model = CustomNet()
    loaded_model.load_state_dict(torch.load(save_path))
    
    # Compare parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2), "Loaded model parameters should match original"