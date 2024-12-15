import torch
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CIFAR10Dataset:
    def __init__(self, root="./data", train=True, download=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        
        # Calculate dataset mean and std
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        
        # Define transformations
        if train:
            self.transform = A.Compose([
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
                    min_width=1,
                    fill_value=self.mean,
                    mask_fill_value=None,
                    p=0.5
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        image = self.transform(image=image)["image"]
        return image, label
    
    def __len__(self):
        return len(self.dataset)

def get_dataloaders(batch_size=128, num_workers=4):
    train_dataset = CIFAR10Dataset(train=True, download=True)
    test_dataset = CIFAR10Dataset(train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 