import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import models

from networks import OriginNetwork, SubNetwork

train_dataset = torchvision.datasets.CIFAR10('./cifar10_data/', train=True, transform=None, download=True)
test_dataset = torchvision.CIFAR10('./cifar10_data/', train=False, transform=None, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



Originmodel = OriginNetwork(models.resnet50(pretrained=False, num_classes=10), num_classes=10)
Submodel = SubNetwork(models.resnet50(pretrained=False, num_classes=10), num_classes=10)



for batch, target in train_loader:


