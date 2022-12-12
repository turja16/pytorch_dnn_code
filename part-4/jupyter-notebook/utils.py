import torch


# Train the model
def train(epoch, model, device, train_loader, optimizer, criterion, input_size, channels):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, input_size*input_size*channels)
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
def test(model, device, test_loader, criterion, input_size, channels):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, input_size*input_size*channels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) '.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))