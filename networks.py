import torch
import torch.nn as nn

class OriginNetwork(nn.Module):
    def __init__(self, model, num_classes):
        super(OriginNetwork, self).__init__()
        self.model = model
        self.num_classes = num_classes
        # init weight

    def forward(self, x):
        x = self.model(x)
        return x


class SubNetwork(nn.Module):
    def __init__(self, model, num_classes):
        super(SubNetwork, self).__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        return x