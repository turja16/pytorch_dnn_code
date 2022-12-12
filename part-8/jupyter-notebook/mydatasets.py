import torch
from torchvision import datasets, transforms


def get_dataset(dataset_name, batch_size):
        # decide which dataset to use
    if dataset_name == 'mnist':
        train_loader, test_loader = load_mnist(batch_size=batch_size)
    elif dataset_name == 'fashion_mnist':
        train_loader, test_loader = load_fashion_mnist(batch_size=batch_size)
    elif dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10(batch_size=batch_size)
    elif dataset_name == 'cifar100':
        train_loader, test_loader = load_cifar100(batch_size=batch_size)
    else:
        print('Invalid dataset')
        exit()
    return train_loader, test_loader

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