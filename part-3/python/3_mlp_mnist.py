import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse

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

# Load the training data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                     transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
    batch_size=64, shuffle=False)    

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

# Train the model
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

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


# main python function
if __name__ == '__main__':

    # parse the command line arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    # argument for training only
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    
    # argument for testing only
    parser.add_argument('--test', action='store_true', default=False,
                        help='test the model')
    
    args = parser.parse_args()

    # train the model
    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test()
        # save the model
        torch.save(model.state_dict(), "mlp_mnist.pt")

    # test the model
    if args.test:
        model.load_state_dict(torch.load("mlp_mnist.pt"))
        test()
        
    


    


