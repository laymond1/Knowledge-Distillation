from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

__all__ = ['distill_loss']

DistillationLossWeights = namedtuple('DistillationLossWeights',
                                     ['distill', 'subnet', 'origin'])
                     
class distill_loss(nn.Module):
    def __init__(self, origin_model, subnet_model, 
                temperature=1.0, loss_weights=DistillationLossWeights(0.5,0.5,0.)):
        super(distill_loss, self).__init__()
        self.origin_model = origin_model
        self.subnet_model = subnet_model
        self.temperature = temperature
        self.loss_weights = loss_weights

        self.last_origin_logits = None
        self.last_subnet_logits = None

        self.loss_distill = None
        self.loss_origin = None
        self.loss_subnet = None

    def forward(self, batch, target):

        if self.loss_weights.origin == 0:
            with torch.no_grad():
                self.last_origin_logits = self.origin_model(batch)
        else:
            self.last_origin_logits = self.origin_model(batch)

        self.last_subnet_logits = self.subnet_model(batch)

        soft_targets = F.log_softmax(self.last_origin_logits / self.temperature, dim=1)
        soft_preds = F.log_softmax(self.last_subnet_logits / self.temperature, dim=1)
        preds = F.log_softmax(self.last_subnet_logits, dim=1)

        self.loss_subnet = F.cross_entropy(self.last_subnet_logits, target)
        # self.loss_subnet = F.kl_div(preds, target) 
        self.loss_distill = F.kl_div(soft_preds, soft_targets.detach(), reduction='batchmean')
        loss = self.loss_subnet * self.loss_weights.subnet + self.loss_distill * self.loss_weights.distill
        
        return loss, preds


            
