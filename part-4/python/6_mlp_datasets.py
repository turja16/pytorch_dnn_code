import torch
from torch import nn
import torch.nn.functional as F
import argparse
from utils import test, train 
from mydatasets import load_mnist, load_fashion_mnist, load_cifar10, load_cifar100

# enable GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


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
    
    # argument to print the model
    parser.add_argument('--print', action='store_true', default=False,
                        help='print the model')
    
    # argument to print the model and trainable parameters
    parser.add_argument('--info', action='store_true', default=False,
                        help='print the model info')
    
    # argument to name the trained model
    parser.add_argument('--name', type=str, default='mnist.pt',
                        help='name of the trained model')

    # argument to decide the dataset
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use: mnist or fashion_mnist')

    # argument to decide the batch size
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training and testing')
    
    
    # parse the arguments
    args = parser.parse_args()


    # decide which dataset to use
    if args.dataset == 'mnist':
        train_loader, test_loader = load_mnist(batch_size=args.batch_size)
        input_size = 28
        channels = 1
        output_size = 10
    elif args.dataset == 'fashion_mnist':
        train_loader, test_loader = load_fashion_mnist(batch_size=args.batch_size)
        input_size = 28
        channels = 1
        output_size = 10
    elif args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10(batch_size=args.batch_size)
        input_size = 32
        channels = 3
        output_size = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader = load_cifar100(batch_size=args.batch_size)
        input_size = 32
        channels = 3
        output_size = 100
    else:
        print('Invalid dataset')
        exit()


        # Build a simple MLP to train on MNIST
    model = nn.Sequential(
        nn.Linear(input_size*input_size*channels, 512),
        nn.ReLU(),
        nn.Linear(512, output_size),
        # nn.ReLU(),
        # nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    ).to(device)


    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Define the loss function
    criterion = nn.NLLLoss()


    # train the model
    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(epoch, model, device, train_loader, optimizer, criterion, input_size, channels)
            test(model, device, test_loader, criterion, input_size, channels)
        # save the model
        torch.save(model.state_dict(), args.name)

    # test the model
    if args.test:
        model.load_state_dict(torch.load(args.name))
        test(model, device, test_loader, criterion, input_size, channels)
        
    # print the model
    if args.print:
        print(model)

    # print the model info
    if args.info:
        print(model)
        print('Model Info')
        for name, param in model.named_parameters():
            print(name, param.shape)
        print('Trainable parameters:\n',sum(p.numel() for p in model.parameters() if p.requires_grad))
    


    


