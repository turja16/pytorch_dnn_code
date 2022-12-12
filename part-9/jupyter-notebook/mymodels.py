import torch

# build a parameterized CNN model for cifar10 and cifar100
# the number of convolutional layers is defined by the parameter num_conv_layers
# the number of filters in each convolutional layer is defined by the parameter num_filters
# the number of fully connected layers is defined by the parameter num_fc_layers
# the number of neurons in each fully connected layer is defined by the parameter num_neurons
# the number of neurons in the last fully connected layer is defined by the parameter num_neurons_last
def CNN_cifar(input_size, channels, output_size, num_conv_layers, num_filters, num_fc_layers, num_neurons, num_neurons_last):
    layers = []
    layers.append(torch.nn.Conv2d(channels, num_filters, kernel_size=3, padding=1))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.MaxPool2d(2))
    for i in range(num_conv_layers-1):
        layers.append(torch.nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(2))
    layers.append(torch.nn.Flatten())
    for i in range(num_fc_layers):
        layers.append(torch.nn.Linear(num_filters*input_size//(2**(num_conv_layers+1))*input_size//(2**(num_conv_layers+1)), num_neurons))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(num_neurons, num_neurons_last))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(num_neurons_last, output_size))
    layers.append(torch.nn.LogSoftmax(dim=1))
    return torch.nn.Sequential(*layers)




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

