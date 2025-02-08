from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

batch_size = 512

train_dataset = FashionMNIST(root='./data', train=True,
                             download=True,
                             transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = FashionMNIST(root='./data', train=False,
                            download=True,
                            transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
