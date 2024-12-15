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

@pytest.fixture
def setup_training():
    model = CustomNet()
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy dataloaders
    train_dataset = DummyDataset(100)
    test_dataset = DummyDataset(50)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    
    return model, device, train_loader, test_loader, criterion, optimizer

def test_train_step(setup_training):
    model, device, train_loader, _, criterion, optimizer = setup_training
    
    # Run one epoch of training
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch=1)
    
    assert isinstance(train_loss, float), "Training loss should be a float"
    assert isinstance(train_acc, float), "Training accuracy should be a float"
    assert 0 <= train_loss <= 10, "Training loss should be reasonable"
    assert 0 <= train_acc <= 100, "Training accuracy should be between 0 and 100"

def test_test_step(setup_training):
    model, device, _, test_loader, criterion, _ = setup_training
    
    # Run test step
    test_loss, test_acc = test(model, device, test_loader, criterion)
    
    assert isinstance(test_loss, float), "Test loss should be a float"
    assert isinstance(test_acc, float), "Test accuracy should be a float"
    assert 0 <= test_loss <= 10, "Test loss should be reasonable"
    assert 0 <= test_acc <= 100, "Test accuracy should be between 0 and 100"

def test_model_training_improves():
    # Create a very small dataset where we expect the model to overfit
    train_dataset = DummyDataset(32)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    
    model = CustomNet()
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Get initial loss
    initial_loss, _ = train(model, device, train_loader, criterion, optimizer, epoch=1)
    
    # Train for a few epochs
    for epoch in range(2, 4):
        final_loss, _ = train(model, device, train_loader, criterion, optimizer, epoch)
    
    assert final_loss < initial_loss, "Loss should decrease during training"

def test_learning_rate_scheduling():
    model = CustomNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Simulate plateau in accuracy
    for _ in range(6):
        scheduler.step(50.0)  # Same accuracy for 6 steps
    
    # Check if learning rate was reduced
    final_lr = optimizer.param_groups[0]['lr']
    assert final_lr == initial_lr * 0.5, "Learning rate should be halved after plateau"

def test_model_saving(tmp_path):
    model = CustomNet()
    save_path = os.path.join(tmp_path, "test_model.pth")
    
    # Save model
    torch.save(model.state_dict(), save_path)
    assert os.path.exists(save_path), "Model file should exist after saving"
    
    # Load model
    new_model = CustomNet()
    new_model.load_state_dict(torch.load(save_path))
    
    # Compare parameters
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2), "Loaded model parameters should match original" 