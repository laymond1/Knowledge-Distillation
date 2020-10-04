import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from networks import OriginNetwork, SubNetwork
from train_test import train, test
from tensorboardX import SummaryWriter
from utils import distill_loss, accuracy, AverageMeter, Logger, savefig, get_mean_and_std

def main(
    arch = 'resnet50',
    num_epochs = 100,
    batch_size = 256,
    checkpoint = 'checkpoint'):

    # Data loading code
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    # train_mean, train_std = get_mean_and_std(torchvision.datasets.CIFAR10('./cifar10_data', train=True, download=False, transform=transform))
    # test_mean, test_std = get_mean_and_std(torchvision.datasets.CIFAR10('./cifar10_data', train=False, download=False, transform=transform))
    # print(train_mean, train_std, test_mean, test_std)

    data_transform = {
        'train' : transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(train_mean, train_std),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'test' : transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(test_mean, test_std),
            transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2020, 0.1991, 0.2011)),
        ]) }

    train_dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=True, transform=data_transform['train'], download=False)
    test_dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=False, transform=data_transform['test'], download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model loading code
    Originmodel = OriginNetwork(models.resnet50(pretrained=False, num_classes=10), num_classes=10)
    Submodel = SubNetwork(models.resnet50(pretrained=False, num_classes=10), num_classes=10)

    # Selecting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Originmodel.to(device)
    Submodel.to(device)

    # Loss & Optimizer
    criterion = distill_loss(origin_model=Originmodel, subnet_model=Submodel, temperature=5)
    optimizer = torch.optim.SGD(Submodel.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Visualization
    writer = SummaryWriter(os.path.join(checkpoint, 'logs'))
    # Logging Metrics
    logger = Logger(os.path.join(checkpoint, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc', 'Test Acc'])

    best_prec1 = 0
    for epoch in range(num_epochs):
        print('')

        # Train : Distilling knowledge to Subnet from origin
        train_loss, train_acc = train(train_loader, Originmodel, Submodel, criterion, optimizer, epoch, device)

        # Test
        test_loss, test_acc = test(test_loader, Originmodel, Submodel, criterion, epoch, device)

        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([lr, train_loss, test_loss, train_acc, test_acc])

        # visualize training process using tensorboardx
        writer.add_scalar('learning rate', lr)
        writer.add_scalars('loss', {'train loss':train_loss, 'test loss':test_loss}, epoch+1)
        writer.add_scalars('accuracy', {'train accuracy':train_acc, 'test accuracy':test_acc}, epoch+1)

        is_best = test_acc > best_prec1 
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint(state={
            'epoch':epoch+1,
            'arch':arch,
            'state_dict':Submodel.state_dict(),
            'best_prec1':best_prec1,
            'optimizer':optimizer.state_dict(),
            'loss':criterion.state_dict()
        }, is_best=is_best, checkpoint=checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint, 'log.jpg'))
    writer.close()
    print(f'Best Accuracy : {best_prec1}')


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


if __name__ == '__main__':
    main()