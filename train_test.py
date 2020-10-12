from __future__ import absolute_import

import torch
from utils import accuracy, AverageMeter


def train(train_loader, Originmodel, Submodel, criterion, optimizer, epoch, device):
    # Train : Distilling knowledge to Subnet from origin
    losses = AverageMeter() 
    top1 = AverageMeter() 
    top5 = AverageMeter()

    Originmodel.train()
    Submodel.train()

    for n_step, (batch, target) in enumerate(train_loader):
        batch, target = batch.to(device), target.to(device)

        loss, preds = criterion(batch, target)
        prec1, prec5 = accuracy(preds, target, topk=(1,5))
        losses.update(loss.item(), batch.size(0))
        top1.update(prec1.item(), batch.size(0))
        top5.update(prec5.item(), batch.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n_step % 20 == 0:
            print('Train Epoch : {} [{}/{} ({:.1f}%)] \tLoss : {:.4f} Top1 : {:.3f}'.format(
                epoch, (n_step+1)*len(batch), len(train_loader.dataset), 
                100.*(n_step+1)/len(train_loader), losses.avg, top1.avg))
    return (losses.avg, top1.avg)

def test(test_loader, Originmodel, Submodel, criterion, epoch, device):
    # Test
    losses = AverageMeter() 
    top1 = AverageMeter() 
    top5 = AverageMeter()

    Originmodel.eval()
    Submodel.eval()  

    with torch.no_grad():
        for batch, target in test_loader:
            batch, target = batch.to(device), target.to(device)

            loss, preds = criterion(batch, target)
            prec1, prec5 = accuracy(preds, target, topk=(1,5))
            losses.update(loss.item(), batch.size(0))
            top1.update(prec1.item(), batch.size(0))
            top5.update(prec5.item(), batch.size(0))

        print('\nTest : Loss : {:.4f} Top1 : {:.4f}'.format(
            losses.avg, top1.avg))
    return (losses.avg, top1.avg)

def train_solo(train_loader, model, criterion, optimizer, epoch, device):
    # Train : Distilling knowledge to Subnet from origin
    losses = AverageMeter() 
    top1 = AverageMeter() 
    top5 = AverageMeter()

    model.train()

    for n_step, (batch, target) in enumerate(train_loader):
        batch, target = batch.to(device), target.to(device)

        loss, preds = criterion(batch, target)
        prec1, prec5 = accuracy(preds, target, topk=(1,5))
        losses.update(loss.item(), batch.size(0))
        top1.update(prec1.item(), batch.size(0))
        top5.update(prec5.item(), batch.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n_step % 20 == 0:
            print('Train Epoch : {} [{}/{} ({:.1f}%)] \tLoss : {:.4f} Top1 : {:.3f}'.format(
                epoch, (n_step+1)*len(batch), len(train_loader.dataset), 
                100.*(n_step+1)/len(train_loader), losses.avg, top1.avg))
    return (losses.avg, top1.avg)


def test_solo(test_loader, model, device):
    model.eval()
    correct = 0
    # test_acc = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            if device is not None:
                data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # prec1, _ = accuracy(data, target, topk=(1,5))

            # test_acc.update(prec1.item(), data.size(0))

        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))