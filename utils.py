import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(
            f'Epoch: {epoch} Loss: {loss.item():0.4f} '
            f'Acc: {100*correct/processed:0.2f}%'
        )
    
    return train_loss/len(train_loader), 100*correct/processed

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n'
    )
    return test_loss, accuracy

def plot_metrics(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(test_losses, label='Test Loss')
    axs[0].set_title('Loss vs Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    axs[1].plot(train_acc, label='Train Accuracy')
    axs[1].plot(test_acc, label='Test Accuracy')
    axs[1].set_title('Accuracy vs Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close() 