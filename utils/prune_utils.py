import copy
import pytorchcv
import torch
import math
import torch.nn as nn
from models.imagenet.mobilenetv2 import InvertedResidual
import models.imagenet.resnet as res

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)
    
def l1_pruning_with_prs_for_mobilenet_v2(model, prs):
    pm = copy.deepcopy(model)
    for n,m in pm.named_modules():
        if isinstance(m, InvertedResidual) and n in prs.keys():
            pr = prs[n]
            num_keep = math.ceil(m.conv[0][0].out_channels * (1-pr))
            l1 = m.conv[0][0].weight.data.abs().sum(dim=(1,2,3))
            _,sorted_indices = torch.sort(l1,descending=True)
            selected_indices = sorted_indices[:num_keep]
            mask_keep = torch.zeros_like(sorted_indices, dtype=torch.bool)
            mask_keep[selected_indices] = True
            # pruned conv1
            pruned_conv1 = nn.Conv2d(in_channels=m.conv[0][0].in_channels, out_channels=num_keep, kernel_size=m.conv[0][0].kernel_size,
                stride=m.conv[0][0].stride, padding=m.conv[0][0].padding, dilation=m.conv[0][0].dilation, 
                groups=m.conv[0][0].groups,
                bias = True if m.conv[0][0].bias != None else False, padding_mode=m.conv[0][0].padding_mode)
            if m.conv[0][0].bias != None:
                pruned_conv1.bias.data = m.conv[0][0].bias.data[mask_keep]
            # pruned conv2
            pruned_conv2 = nn.Conv2d(in_channels=num_keep, out_channels=num_keep, kernel_size=m.conv[1][0].kernel_size,
                stride=m.conv[1][0].stride, padding=m.conv[1][0].padding, dilation=m.conv[1][0].dilation, 
                groups=num_keep,
                bias = True if m.conv[1][0].bias != None else False, padding_mode=m.conv[1][0].padding_mode)
            if m.conv[1][0].bias != None:
                pruned_conv2.bias.data = m.conv[1][0].bias.data
            # pruned conv3
            pruned_conv3 = nn.Conv2d(in_channels=num_keep, out_channels=m.conv[2].out_channels, kernel_size=m.conv[2].kernel_size,
                stride=m.conv[2].stride, padding=m.conv[2].padding, dilation=m.conv[2].dilation, 
                groups=m.conv[2].groups,
                bias = True if m.conv[2].bias != None else False, padding_mode=m.conv[2].padding_mode)
            if m.conv[2].bias != None:
                pruned_conv3.bias.data = m.conv[2].bias.data
            # pruned bn1
            pruned_bn1 = nn.BatchNorm2d(num_features=num_keep, eps=m.conv[0][1].eps, momentum=m.conv[0][1].momentum,
                affine=m.conv[0][1].affine,track_running_stats=m.conv[0][1].track_running_stats)
            # pruned bn2
            pruned_bn2 = nn.BatchNorm2d(num_features=num_keep, eps=m.conv[1][1].eps, momentum=m.conv[1][1].momentum,
                affine=m.conv[1][1].affine,track_running_stats=m.conv[1][1].track_running_stats)
            # update paras
            pruned_conv1.weight.data = m.conv[0][0].weight.data[mask_keep]
            pruned_conv2.weight.data = m.conv[1][0].weight.data[mask_keep, :, :, :]
            pruned_conv3.weight.data = m.conv[2].weight.data[ :, mask_keep, :, :]
            pruned_bn1.weight.data = m.conv[0][1].weight.data[mask_keep]
            pruned_bn1.bias.data = m.conv[0][1].bias.data[mask_keep]
            pruned_bn1.running_mean.data = m.conv[0][1].running_mean.data[mask_keep]
            pruned_bn1.running_var.data = m.conv[0][1].running_var.data[mask_keep]
            pruned_bn2.weight.data = m.conv[1][1].weight.data[mask_keep]
            pruned_bn2.bias.data = m.conv[1][1].bias.data[mask_keep]
            pruned_bn2.running_mean.data = m.conv[1][1].running_mean.data[mask_keep]
            pruned_bn2.running_var.data = m.conv[1][1].running_var.data[mask_keep]
            # replace modules
            _set_module(pm,n+'.conv.0.0',pruned_conv1)
            _set_module(pm,n+'.conv.0.1',pruned_bn1)
            _set_module(pm,n+'.conv.1.0',pruned_conv2)
            _set_module(pm,n+'.conv.1.1',pruned_bn2)
            _set_module(pm,n+'.conv.2',pruned_conv3)
    return pm
  
def l1_pruning_with_prs_for_resnet34(model, prs):
    pm = copy.deepcopy(model)
    for n,m in pm.named_modules():
        if isinstance(m, res.BasicBlock) and n in prs.keys():
            pr = prs[n]
            num_keep = math.ceil(m.conv1.out_channels * (1-pr))
            l1 = m.conv1.weight.data.abs().sum(dim=(1,2,3))
            _,sorted_indices = torch.sort(l1,descending=True)
            selected_indices = sorted_indices[:num_keep]
            mask_keep = torch.zeros_like(sorted_indices, dtype=torch.bool)
            mask_keep[selected_indices] = True
            # pruned conv1
            pruned_conv1 = nn.Conv2d(in_channels=m.conv1.in_channels, out_channels=num_keep, kernel_size=m.conv1.kernel_size,
                stride=m.conv1.stride, padding=m.conv1.padding, dilation=m.conv1.dilation, 
                groups=m.conv1.groups,
                bias = True if m.conv1.bias != None else False, padding_mode=m.conv1.padding_mode)
            if m.conv1.bias != None:
                pruned_conv1.bias.data = m.conv1.bias.data[mask_keep]
            # pruned conv2
            pruned_conv2 = nn.Conv2d(in_channels=num_keep, out_channels=m.conv2.out_channels, kernel_size=m.conv2.kernel_size,
                stride=m.conv2.stride, padding=m.conv2.padding, dilation=m.conv2.dilation, 
                groups=m.conv2.groups,
                bias = True if m.conv2.bias != None else False, padding_mode=m.conv2.padding_mode)
            if m.conv2.bias != None:
                pruned_conv2.bias.data = m.conv2.bias.data
            # pruned bn1
            pruned_bn1 = nn.BatchNorm2d(num_features=num_keep, eps=m.bn1.eps, momentum=m.bn1.momentum,
                affine=m.bn1.affine,track_running_stats=m.bn1.track_running_stats)
            # update paras
            pruned_conv1.weight.data = m.conv1.weight.data[mask_keep]
            pruned_conv2.weight.data = m.conv2.weight.data[:, mask_keep, :, :]
            pruned_bn1.weight.data = m.bn1.weight.data[mask_keep]
            pruned_bn1.bias.data = m.bn1.bias.data[mask_keep]
            pruned_bn1.running_mean.data = m.bn1.running_mean.data[mask_keep]
            pruned_bn1.running_var.data = m.bn1.running_var.data[mask_keep]
            # replace modules
            _set_module(pm,n+'.conv1',pruned_conv1)
            _set_module(pm,n+'.bn1',pruned_bn1)
            _set_module(pm,n+'.conv2',pruned_conv2)
    return pm

def l1_pruning_with_prs_for_fcn_resnet101(model, prs, prune_single_conv = True):
    pm = copy.deepcopy(model)
    for n,m in pm.named_modules():
        if isinstance(m, pytorchcv.models.resnet.ResBottleneck) and n in prs.keys():
            pr = prs[n]
            if not prune_single_conv:
                ## conv1 l1
                num_keep_conv1 = math.ceil(m.conv1.conv.out_channels * (1-pr))
                l1_conv1 = m.conv1.conv.weight.data.abs().sum(dim=(1,2,3))
                _,sorted_indices = torch.sort(l1_conv1,descending=True)
                selected_indices_conv1 = sorted_indices[:num_keep_conv1]
                mask_keep_conv1 = torch.zeros_like(sorted_indices, dtype=torch.bool)
                mask_keep_conv1[selected_indices_conv1] = True
            # conv2 l1
            num_keep_conv2 = math.ceil(m.conv2.conv.out_channels * (1-pr))
            l1_conv2 = m.conv2.conv.weight.data.abs().sum(dim=(1,2,3))
            _,sorted_indices = torch.sort(l1_conv2,descending=True)
            selected_indices_conv2 = sorted_indices[:num_keep_conv2]
            mask_keep_conv2 = torch.zeros_like(sorted_indices, dtype=torch.bool)
            mask_keep_conv2[selected_indices_conv2] = True
            if not prune_single_conv:
                # pruned conv1
                pruned_conv1 = nn.Conv2d(in_channels=m.conv1.conv.in_channels, out_channels=num_keep_conv1, kernel_size=m.conv1.conv.kernel_size,
                    stride=m.conv1.conv.stride, padding=m.conv1.conv.padding, dilation=m.conv1.conv.dilation, 
                    groups=m.conv1.conv.groups,
                    bias = True if m.conv1.conv.bias != None else False, padding_mode=m.conv1.conv.padding_mode)
                if m.conv1.conv.bias != None:
                    pruned_conv1.bias.data = m.conv1.conv.bias.data[mask_keep_conv1]
                # pruned conv2
                pruned_conv2 = nn.Conv2d(in_channels=num_keep_conv1, out_channels=num_keep_conv2, kernel_size=m.conv2.conv.kernel_size,
                    stride=m.conv2.conv.stride, padding=m.conv2.conv.padding, dilation=m.conv2.conv.dilation, 
                    groups=m.conv2.conv.groups,
                    bias = True if m.conv2.conv.bias != None else False, padding_mode=m.conv2.conv.padding_mode)
            else:
                pruned_conv2 = nn.Conv2d(in_channels=m.conv2.conv.in_channels, out_channels=num_keep_conv2, kernel_size=m.conv2.conv.kernel_size,
                    stride=m.conv2.conv.stride, padding=m.conv2.conv.padding, dilation=m.conv2.conv.dilation, 
                    groups=m.conv2.conv.groups,
                    bias = True if m.conv2.conv.bias != None else False, padding_mode=m.conv2.conv.padding_mode)
            if m.conv2.conv.bias != None:
                pruned_conv2.bias.data = m.conv2.conv.bias.data[mask_keep_conv2]
            # pruned conv3
            pruned_conv3 = nn.Conv2d(in_channels=num_keep_conv2, out_channels=m.conv3.conv.out_channels, kernel_size=m.conv3.conv.kernel_size,
                stride=m.conv3.conv.stride, padding=m.conv3.conv.padding, dilation=m.conv3.conv.dilation, 
                groups=m.conv3.conv.groups,
                bias = True if m.conv3.conv.bias != None else False, padding_mode=m.conv3.conv.padding_mode)
            if m.conv3.conv.bias != None:
                pruned_conv3.bias.data = m.conv3.conv.bias.data
            if not prune_single_conv:
                # pruned bn1
                pruned_bn1 = nn.BatchNorm2d(num_features=num_keep_conv1, eps=m.conv1.bn.eps, momentum=m.conv1.bn.momentum,
                    affine=m.conv1.bn.affine,track_running_stats=m.conv1.bn.track_running_stats)
            # pruned bn2
            pruned_bn2 = nn.BatchNorm2d(num_features=num_keep_conv2, eps=m.conv2.bn.eps, momentum=m.conv2.bn.momentum,
                affine=m.conv2.bn.affine,track_running_stats=m.conv2.bn.track_running_stats)
            # update paras
            if not prune_single_conv:
                pruned_conv1.weight.data = m.conv1.conv.weight.data[mask_keep_conv1]
                pruned_bn1.weight.data = m.conv1.bn.weight.data[mask_keep_conv1]
                pruned_bn1.bias.data = m.conv1.bn.bias.data[mask_keep_conv1]
                pruned_bn1.running_mean.data = m.conv1.bn.running_mean.data[mask_keep_conv1]
                pruned_bn1.running_var.data = m.conv1.bn.running_var.data[mask_keep_conv1]
                temp = m.conv2.conv.weight.data[:, mask_keep_conv1, :, :]
                pruned_conv2.weight.data = temp[mask_keep_conv2]
            else:
                pruned_conv2.weight.data = m.conv2.conv.weight.data[mask_keep_conv2]
            pruned_conv3.weight.data = m.conv3.conv.weight.data[:, mask_keep_conv2, :, :]
            pruned_bn2.weight.data = m.conv2.bn.weight.data[mask_keep_conv2]
            pruned_bn2.bias.data = m.conv2.bn.bias.data[mask_keep_conv2]
            pruned_bn2.running_mean.data = m.conv2.bn.running_mean.data[mask_keep_conv2]
            pruned_bn2.running_var.data = m.conv2.bn.running_var.data[mask_keep_conv2]
            # replace modules
            if not prune_single_conv:
                _set_module(pm,n+'.conv1.conv',pruned_conv1)
                _set_module(pm,n+'.conv1.bn',pruned_bn1)
            _set_module(pm,n+'.conv2.conv',pruned_conv2)
            _set_module(pm,n+'.conv2.bn',pruned_bn2)
            _set_module(pm,n+'.conv3.conv',pruned_conv3)
    del model
    torch.cuda.empty_cache()
    return pm
            