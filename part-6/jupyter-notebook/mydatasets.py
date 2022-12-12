import torch
from torchvision import datasets, transforms


# function to load the MNIST dataset
def load_mnist(batch_size=64, shuffle=True):
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])),
        batch_size=batch_size, shuffle=shuffle)    

    # Load the test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])),
        batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader


# function to load the Fashion-MNIST dataset
def load_fashion_mnist(batch_size=64, shuffle=True):
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])),
        batch_size=batch_size, shuffle=shuffle)    

    # Load the test data
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])),
        batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader

# function to load the CIFAR10 dataset
def load_cifar10(batch_size=64, shuffle=True):
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])),
        batch_size=batch_size, shuffle=shuffle)    

    # Load the test data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])),
        batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader

# function to load the CIFAR100 dataset
def load_cifar100(batch_size=64, shuffle=True):
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])),
        batch_size=batch_size, shuffle=shuffle)    

    # Load the test data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])),
        batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader