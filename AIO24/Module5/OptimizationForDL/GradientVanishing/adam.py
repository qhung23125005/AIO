import torch.nn as nn
import torch.optim as optim
from model import MLP
from setup_device import device
from data_processor import train_loader, test_loader
from visualizer import visualize_loss_acc
from train_model import fit

def main():
    num_epochs = 300
    lr = 0.01
    model = MLP(input_dims=784, hidden_dims=128, output_dims=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, train_acc, val_losses, val_acc = fit(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        test_loader, 
        num_epochs, 
        device
    )

    visualize_loss_acc(train_losses, train_acc, val_losses, val_acc, 'Adam')

if __name__ == '__main__':
    main()