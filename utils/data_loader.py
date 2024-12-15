import torch
from torchvision import datasets
import numpy as np

class AlbumentationsDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def get_dataloaders(train_transform, test_transform, batch_size=128):
    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    train_dataset = AlbumentationsDataset(trainset, train_transform)
    test_dataset = AlbumentationsDataset(testset, test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader 