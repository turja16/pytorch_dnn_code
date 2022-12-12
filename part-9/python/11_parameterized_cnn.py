import torch
from torch import nn
import torch.nn.functional as F
import argparse
from utils import test, train_keep_best 
from mydatasets import get_dataset
from mymodels import MLP, CNN, CNN_cifar
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


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
    
    # argument to resume training
    parser.add_argument('--resume', action='store_true', default=False,
                        help='learning rate for training')
    
    # argument to get image input from command line to classify
    parser.add_argument('--image', type=str, default=None,
                        help='image file to classify')
    
    # argument to decide the number of convolutional layers
    parser.add_argument('--conv_layers', type=int, default=3,
                        help='number of convolutional layers')
    
    # argument to decide the number of filers in each convolutional layer
    parser.add_argument('--filters', type=int, default=32,
                        help='number of filers in each convolutional layer')
    
    # argument to decide the number of hidden layers
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='number of hidden layers')
    
    # argument to decide the number of neuron in each hidden layer
    parser.add_argument('--neurons', type=int, default=128,
                        help='number of neuron in each hidden layer')
    
    # argument to decide the number of neurons in the last fully connected layer
    parser.add_argument('--neurons_last', type=int, default=128,
                        help='number of neurons in the last fully connected layer')

   
    # parse the arguments
    args = parser.parse_args()
    
    # decide which dataset to use
    if args.dataset == 'mnist':
        input_size = 28
        channels = 1
        output_size = 10
    elif args.dataset == 'fashion_mnist':
        input_size = 28
        channels = 1
        output_size = 10
    elif args.dataset == 'cifar10':
        input_size = 32
        channels = 3
        output_size = 10
    elif args.dataset == 'cifar100':
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
    elif args.model == 'cnn_cifar':
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            model = CNN_cifar(input_size, channels, output_size, args.conv_layers, args.filters, args.hidden_layers, args.neurons, args.neurons_last).to(device)
    else:
        print('Invalid model')
        exit()

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Define the loss function
    criterion = nn.NLLLoss()


    # train the model
    if args.train:
        train_loader, test_loader = get_dataset(args.dataset, args.batch_size) 
        acc_now = 0.0
        for epoch in range(1, args.epochs + 1):
            acc_now = train_keep_best(epoch, model, args.model, device, train_loader, test_loader, optimizer, criterion, input_size, channels, args.name, acc_now)


    if args.test:
        train_loader, test_loader = get_dataset(args.dataset, args.batch_size) 
    # load the model as checkpoint and see multiple information
        checkpoint = torch.load(args.name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']
        print('Model trained for', epoch, 'epochs with accuracy', acc, 'and loss', loss)
        test(model, args.model, device, test_loader, criterion, input_size, channels)

    if args.resume:
        train_loader, test_loader = get_dataset(args.dataset, args.batch_size) 
        # load the model as checkpoint and see multiple information
        checkpoint = torch.load(args.name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']
        print('Model trained for', epoch, 'epochs with accuracy', acc, 'and loss', loss)
        print('Resuming training...')
        for epoch in range(epoch+1, epoch + args.epochs):
            acc = train_keep_best(epoch, model, args.model, device, train_loader, test_loader, optimizer, criterion, input_size, channels, args.name, acc)

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

    if args.image:
        # classes of cifar10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # load the model as checkpoint and see multiple information
        checkpoint = torch.load(args.name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']
        print('Model trained for', epoch, 'epochs with accuracy', acc, 'and loss', loss)
        print('This the image you want to classify:')
        # show the image
        img = Image.open(args.image)
        plt.imshow(img)
        plt.show()
        # transform the image
        transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        
        img = transform(img)
        # show the transformed image
        plt.imshow(img[0], cmap='gray')
        plt.show()
        
        # add a dimension to the image
        img = img.unsqueeze(0)

        # classify the image
        model.eval()
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            print('The image is classified as', pred.item(), 'which is', classes[pred.item()])





    


