from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import os

batch_size = 512

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_file_directory, 'data')

train_dataset = FashionMNIST(root=path, train=True,
                             download=True,
                             transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataset = FashionMNIST(root=path, train=False,
                            download=True,
                            transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
