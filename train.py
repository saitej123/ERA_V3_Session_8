import torch
import torch.nn as nn
import torch.optim as optim
from model import CustomNet
from dataset import get_dataloaders
from utils import train, test, plot_metrics


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model and move it to device
    model = CustomNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Get dataloaders
    train_loader, test_loader = get_dataloaders(batch_size=128)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training loop
    epochs = 50
    best_acc = 0
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(
            model, device, train_loader, criterion, optimizer, epoch
        )
        test_loss, test_accuracy = test(
            model, device, test_loader, criterion
        )

        # Update learning rate
        scheduler.step(test_accuracy)

        # Save metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

        # Save best model
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        # Plot metrics
        plot_metrics(train_losses, test_losses, train_acc, test_acc)

        # Early stopping if accuracy reaches 85%
        if test_accuracy >= 85:
            print(f"Reached target accuracy of 85% at epoch {epoch}")
            break

    print(f"Best Test Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main() 