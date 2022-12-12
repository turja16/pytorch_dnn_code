import torch


def MLP(input_size, channels, hidden_size, output_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size*input_size*channels, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
        torch.nn.LogSoftmax(dim=1)
    )

def CNN(input_size, channels, output_size):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(128*input_size//8*input_size//8, output_size),
        torch.nn.LogSoftmax(dim=1)
    )

# CNN for mnist and fashion_mnist
def CNN_mnist(input_size, channels, output_size):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(64*input_size//4*input_size//4, output_size),
        torch.nn.LogSoftmax(dim=1)
    )

