import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# enable GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


# Build a simple MLP to train on MNIST
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    # nn.ReLU(),
    # nn.Linear(256, 10),
    nn.LogSoftmax(dim=1)
).to(device)

# Load the test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
    batch_size=64, shuffle=True)


# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the loss function
criterion = nn.NLLLoss()

# Test the model
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 784)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) '.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Load the model
model.load_state_dict(torch.load('mlp_mnist.pt'))

# Test the model
test()

