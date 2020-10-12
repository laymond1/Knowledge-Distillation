import os
import numpy as np
import torch
import torch.nn as nn
import models

from torchsummary import summary

def resnet56_pruning(model, dataset, save, device, v='A'):
    skip = {
        'A': [16, 20, 38, 54],
        'B': [16, 18, 20, 34, 38, 54],
    }

    prune_prob = {
        'A': [0.1, 0.1, 0.1],
        'B': [0.6, 0.3, 0.1],
    }

    layer_id = 1
    cfg = []
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in skip[v]:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                layer_id += 1
                continue
            if layer_id % 2 == 0:
                if layer_id <= 18:
                    stage = 0
                elif layer_id <= 36:
                    stage = 1
                else:
                    stage = 2
                prune_prob_stage = prune_prob[v][stage]
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1,2,3))
                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                cfg.append(num_keep)
                layer_id += 1
                continue
            layer_id += 1

    newmodel =  models.__dict__['resnet'](dataset='cifar10', depth=56, cfg=cfg)
    if device is not None:
        newmodel.to(device)

    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    conv_count = 1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join('./resnet56/0/', 'pruned.pth.tar'))

    # Measure model size & params size
    print("Origin model summary")
    summary(model, input_size=(3, 32, 32), batch_size=1, device='cuda') # CIFAR10 Input size
    print("\nSubnet model summary")
    summary(newmodel, input_size=(3, 32, 32), batch_size=1, device='cuda')
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    print("number of parameters: "+str(num_parameters))
    return newmodel, num_parameters

def resnet56_weight_pruning(model, percent=0.3):
    #pruning 
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()
    conv_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * percent)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))

    torch.save({'state_dict': model.state_dict()}, os.path.join('./resnet56/2/', 'pruned.pth.tar'))

    return model

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