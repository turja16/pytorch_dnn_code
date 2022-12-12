import torch
from torch import nn
import torch.nn.functional as F
import argparse
from utils import test, train_keep_best 
from mydatasets import load_mnist, load_fashion_mnist, load_cifar10, load_cifar100
from mymodels import MLP, CNN

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
    
    # argument to decide the model type
    parser.add_argument('--model', type=str, default='mlp',
                        help='model to use: mlp or cnn')
    
    # argument to decide the batch size
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training and testing')
    
    # argument to decide the learning rate
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for training')

    
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


    # define the model and configure the number of layers
    if args.model == 'mlp':
        hidden_size = 512
        model = MLP(input_size, channels, hidden_size, output_size).to(device)
    elif args.model == 'cnn':
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            model = CNN_mnist(input_size, channels, output_size).to(device)
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            model = CNN(input_size, channels, output_size).to(device)
    else:
        print('Invalid model')
        exit()

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Define the loss function
    criterion = nn.NLLLoss()


    # train the model
    if args.train:
        acc_now = 0.0
        for epoch in range(1, args.epochs + 1):
            acc_now = train_keep_best(epoch, model, args.model, device, train_loader, test_loader, optimizer, criterion, input_size, channels, args.name, acc_now)

    # test the model
    if args.test:
        model.load_state_dict(torch.load(args.name))
        test(model, args.model, device, test_loader, criterion, input_size, channels)
        
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
    


    


