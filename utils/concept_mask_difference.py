import pytorchcv
import torch
from models.imagenet.mobilenetv2 import InvertedResidual
import models.imagenet.resnet as res
import torch.nn.functional as F


def get_featuremap_for_resnet(model, sampled_loader, device):
    model = model.to(device)
    feature_maps_out_id = {}
    feature_maps_in_id = {}

    def hook_fn(module, input, output):
        if id(module) in feature_maps_out_id.keys():
            feature_maps_out_id[id(module)] = torch.cat(
                [feature_maps_out_id[id(module)], output.detach().cpu()], dim=0
            )
            feature_maps_in_id[id(module)] = torch.cat(
                [feature_maps_in_id[id(module)], input[0].detach().cpu()], dim=0
            )
        else:
            feature_maps_out_id[id(module)] = output.detach().cpu()
            feature_maps_in_id[id(module)] = input[0].detach().cpu()

    handles = []
    for _, m in model.named_modules():
        if isinstance(m, res.BasicBlock) or isinstance(m, pytorchcv.models.resnet.ResBottleneck):
            handles.append(m.register_forward_hook(hook_fn))
    for idx, (samples, labels) in enumerate(sampled_loader):
        samples, labels = samples.to(device), labels.to(device)
        with torch.no_grad():
            if 'FCN' in str(type(model)):
                _ = model(samples)
            else:
                _, _ = model(samples, out_feats=True)
        break
    fm_in = {}
    fm_out = {}
    for n, m in model.named_modules():
        if isinstance(m, res.BasicBlock) or isinstance(m, pytorchcv.models.resnet.ResBottleneck):
            fm_out[n] = feature_maps_out_id[id(m)]
            fm_in[n] = feature_maps_in_id[id(m)]
    for handle in handles:
        handle.remove()

    return fm_in, fm_out

def get_featuremap_for_mobilenet_v2(model, sampled_loader, device):
    model = model.to(device)
    feature_maps_out_id = {}
    feature_maps_in_id = {}

    def hook_fn(module, input, output):
        if id(module) in feature_maps_out_id.keys():
            feature_maps_out_id[id(module)] = torch.cat(
                [feature_maps_out_id[id(module)], output.detach().cpu()], dim=0
            )
            feature_maps_in_id[id(module)] = torch.cat(
                [feature_maps_in_id[id(module)], input[0].detach().cpu()], dim=0
            )
        else:
            feature_maps_out_id[id(module)] = output.detach().cpu()
            feature_maps_in_id[id(module)] = input[0].detach().cpu()

    handles = []
    for _, m in model.named_modules():
        if isinstance(m, InvertedResidual) and m.use_res_connect:
            handles.append(m.register_forward_hook(hook_fn))
    for idx, (samples, labels) in enumerate(sampled_loader):
        samples, labels = samples.to(device), labels.to(device)
        with torch.no_grad():
            _, _ = model(samples, out_feats=True)
        break
    fm_in = {}
    fm_out = {}
    for n, m in model.named_modules():
        if isinstance(m, InvertedResidual) and m.use_res_connect:
            fm_out[n] = feature_maps_out_id[id(m)]
            fm_in[n] = feature_maps_in_id[id(m)]
    for handle in handles:
        handle.remove()

    return fm_in, fm_out

def get_mask(featuremap, HW=224, th=0.2):
    with torch.no_grad():
        B, C, H_orig, W_orig = featuremap.shape
        num_elements = H_orig * W_orig
        threshold_index = int(th * (num_elements - 1))
        sorted_abc, _ = torch.sort(featuremap.view(B, C, -1), dim=-1, descending=True)
        Tbc = sorted_abc[:, :, threshold_index]
        del sorted_abc
        torch.cuda.empty_cache() 
        Abc_interpolated = F.interpolate(
            featuremap, size=(HW, HW), mode="bilinear", align_corners=False
        )
        Tbc = Tbc.view(B, C, 1, 1)
        mask_cond = Tbc == 0
        mask1 = Abc_interpolated != 0
        mask2 = Abc_interpolated >= Tbc
        masks = torch.where(mask_cond, mask1, mask2)

    return masks

def normalize_dict(datas, new_min=0.2, new_max=1.0):
    values = list(datas.values())
    old_min = min(values)
    old_max = max(values)        
    return {
        k: ((v - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        for k, v in datas.items()
    }

def get_difference_of_concept_mask_mobilenet_v2(model, mask_fm_in, mask_fm_out, device):
    doc = {}
    model = model.to(device).eval()
    for n, m in model.named_modules():
        if isinstance(m, InvertedResidual) and m.use_res_connect:
            try:
                doc[n] = torch.sum(mask_fm_in[n].to(device) != mask_fm_out[n].to(device)).item()
            except:
                doc[n] = (mask_fm_in[n].cpu() ^ mask_fm_out[n].cpu()).sum().item()
            torch.cuda.empty_cache()
    return doc
    
def get_difference_of_concept_mask_resnet34(model, mask_fm_in, mask_fm_out, device):
    doc = {}
    model = model.to(device).eval()
    for n, m in model.named_modules():
        if isinstance(m, res.BasicBlock) and m.conv1.in_channels == m.conv1.out_channels:
            try:
                doc[n] = torch.sum(mask_fm_in[n].to(device) != mask_fm_out[n].to(device)).item()
            except:
                doc[n] = (mask_fm_in[n].cpu() ^ mask_fm_out[n].cpu()).sum().item()
            torch.cuda.empty_cache()
    return doc

def get_difference_of_concept_mask_fcn_resnet101(model, mask_fm_in, mask_fm_out, device):
    # tabu_layers = ['backbone.layer1.0', 'backbone.layer2.0', 'backbone.layer3.0','backbone.layer4.0', \
    #                    'backbone.layer1.2','backbone.layer2.3','backbone.layer3.5','backbone.layer4.2']
    tabu_layers = []
    doc = {}
    model = model.to(device).eval()
    for n, m in model.named_modules():
        if isinstance(m, pytorchcv.models.resnet.ResBottleneck) and mask_fm_in[n].shape == mask_fm_out[n].shape and n not in tabu_layers:
            try:
                doc[n] = torch.sum(mask_fm_in[n].to(device) != mask_fm_out[n].to(device)).item()
            except:
                doc[n] = (mask_fm_in[n].cpu() ^ mask_fm_out[n].cpu()).sum().item()
            torch.cuda.empty_cache()
    return doc