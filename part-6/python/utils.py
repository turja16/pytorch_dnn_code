import torch


# train the model but keep the best model
def train_keep_best(epoch, model, type, device, train_loader, test_loader, optimizer, criterion, input_size, channels, name, acc):
    best_acc = acc
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if type == 'mlp':
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
                100. * batch_idx / len(train_loader.dataset), loss.item()))

    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if type == 'mlp':
                data = data.view(-1, input_size*input_size*channels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) '.format(
        test_loss, correct, len(test_loader.dataset), acc))
    if acc > best_acc:
        print('Found better model, saving...')
        best_acc = acc
        torch.save(model.state_dict(), name)
    else:
        print('No better model found')
    return best_acc


# Test the model
def test(model, type, device, test_loader, criterion, input_size, channels):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if type == 'mlp':
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