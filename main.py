import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
# import torchvision.models as models
import models

from networks import OriginNetwork, SubNetwork, resnet56_pruning, resnet56_weight_pruning
from train_test import train, test, train_solo, test_solo
from tensorboardX import SummaryWriter
from utils import distill_loss, accuracy, AverageMeter, Logger, savefig, get_mean_and_std, init_params


def main(
    arch = 'resnet_weight',
    num_epochs = 160,
    batch_size = 256,
    load_checkpoint_dir = 'resnet56/2/model_best.pth.tar',
    save_checkpoint_dir = 'resnet56/2/distill/scratch/',
    save_prune = 'resnet56/2/',
    parrallel=True,
    scratch=True):

    # Data loading code
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    # train_mean, train_std = get_mean_and_std(torchvision.datasets.CIFAR10('./cifar10_data', train=True, download=True, transform=transform))
    # test_mean, test_std = get_mean_and_std(torchvision.datasets.CIFAR10('./cifar10_data', train=False, download=True, transform=transform))
    # print(train_mean, train_std, test_mean, test_std)

    data_transform = {
        'train' : transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
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
    # Originmodel = OriginNetwork(models.resnet50(pretrained=False, num_classes=10), num_classes=10)
    # Submodel = SubNetwork(models.resnet50(pretrained=False, num_classes=10), num_classes=10)
    # Originmodel = models.__dict__[arch](dataset='cifar10', depth=56) # Filter pruning
    Originmodel = models.__dict__[arch](num_classes=10, depth=56) # Weight pruning
    # Submodel = models.__dict__[arch](dataset='cifar10', depth=56)

    # Selecting device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    Originmodel.to(device)
    # Submodel.to(device) # resnet56_pruning 에서 device로 옮겨짐.

    # Loading Checkpoint
    load_checkpoint(Originmodel, checkpoint_dir=load_checkpoint_dir, parrallel=parrallel)
    unpruned_acc = test_solo(test_loader, Originmodel, device)
    # Submodel, num_parameters = resnet56_pruning(Originmodel, dataset='cifar10', save=save_prune, device=device, v='A') # Filter
    Submodel = resnet56_weight_pruning(Originmodel, percent=0.3) # Weight
    pruned_acc = test_solo(test_loader, Submodel, device)
    # Pruned before & after accuracy save
    with open(os.path.join(save_prune, "prune.txt"), "w") as fp:
        fp.write("Before pruning Test accuracy : \n"+str(unpruned_acc.item())+"\n")
        # fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(pruned_acc.item())+"\n")

    # Scratch training
    if scratch:
        Submodel.train()
        init_params(Submodel)
        test_solo(test_loader, Submodel, device)

    # Loss & Optimizer
    criterion = distill_loss(origin_model=Originmodel, subnet_model=Submodel, temperature=5)
    optimizer = torch.optim.SGD(Submodel.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,25,30], gamma= 0.1) 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma= 0.1) # Scratch

    # Visualization
    writer = SummaryWriter(os.path.join(save_checkpoint_dir, 'logs'))
    # Logging Metrics
    title = 'ResNet56'
    logger = Logger(os.path.join(save_checkpoint_dir, 'log.txt'), title=title)
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

        # Learning Rate scheduler
        scheduler.step()

        is_best = test_acc > best_prec1 
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint(state={
            'epoch':epoch+1,
            'arch':arch,
            'state_dict':Submodel.state_dict(),
            'best_prec1':best_prec1,
            'optimizer':optimizer.state_dict(),
            'loss':criterion.state_dict()
        }, is_best=is_best, checkpoint=save_checkpoint_dir)

    logger.close()
    logger.plot()
    savefig(os.path.join(save_checkpoint_dir, 'log.jpg'))
    writer.close()
    print(f'Best Accuracy : {best_prec1}')


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))

def load_checkpoint(model, checkpoint_dir, parrallel=False):
    if os.path.isfile(checkpoint_dir):
        print("Loading pretrained model '{}'".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        target_state_dict = checkpoint['state_dict']
        epoch = checkpoint['epoch']
        # arch = checkpoint['arch']
        if parrallel:
            from collections import OrderedDict
            target_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]
                target_state_dict[name] = v
        else:
            best_prec1 = checkpoint['best_prec1']
            optimizer = checkpoint['optimizer']
            print("Loaded checkpoint '{}' \nBest Acc : {:.3f} (epoch {})".format(checkpoint_dir, best_prec1, epoch))
        model.load_state_dict(target_state_dict)

    else:
        print("No Checkpoint found at '{}'".format(checkpoint_dir))



if __name__ == '__main__':
    main()