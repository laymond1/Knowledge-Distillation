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

        if n_step % 10 == 0:
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