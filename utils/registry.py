import torch
from torchvision import datasets, transforms as T
from models.cifar.resnet import resnet8x4, resnet8, resnet14, resnet20, resnet32, resnet32x4, resnet44, resnet56, resnet110
from models.cifar.resnetv2 import resnet18 as resnet18_cifar
from models.cifar.resnetv2 import resnet34 as resnet34_cifar
from models.cifar.vgg import vgg8, vgg8_bn, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from pytorchcv.model_provider import get_model as ptcv_get_model
import models.imagenet.vgg as vgg_im
from models.imagenet.mobilenetv2 import mobilenet_v2
from models.imagenet.resnet import resnet34, resnet50
import os
import torch.nn as nn 
from torchvision import models

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    'cub200':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_dogs':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_cars':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_64x64': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'tiny_imagenet': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'imagenet_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    
    # for semantic segmentation
    'camvid': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'nyuv2': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}
IMAGENET_MODEL_DICT = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'mobilenet_v2': mobilenet_v2,
    'vgg8': vgg_im.vgg8,
    'vgg8_bn': vgg_im.vgg8_bn,
    'vgg11': vgg_im.vgg11,
    'vgg11_bn': vgg_im.vgg11_bn,
    'vgg13': vgg_im.vgg13,
    'vgg13_bn': vgg_im.vgg13_bn,
    'vgg16': vgg_im.vgg16,
    'vgg16_bn': vgg_im.vgg16_bn,
    'vgg19': vgg_im.vgg19,
    'vgg19_bn': vgg_im.vgg19_bn,
}
ELSE_MODEL_DICT = {
    'resnet18': resnet18_cifar,
    'resnet34': resnet34_cifar,
    'mobilenet_v2': mobilenet_v2,
    'resnet8x4': resnet8x4,
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet32x4': resnet32x4,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'vgg8': vgg8,
    'vgg8_bn': vgg8_bn,
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,
}

def get_model(name: str, num_classes, dataset, pretrained=True):
    if dataset == 'imagenet':
        model = IMAGENET_MODEL_DICT[name](pretrained=pretrained)
    elif name == 'fcn_resnet101':
        model = ptcv_get_model("fcn8sd_resnetd101b_voc", pretrained=False)
        para = torch.load('models/fcn8sd_resnetd101b_voc-8040-66edc0b0.pth')
        if pretrained:
            old_keys = list(para.keys())
            for old_key in old_keys:
                if "init_block" in old_key:
                    new_key = old_key.replace("init_block", "0")
                    para[new_key] = para.pop(old_key)
                elif "stage1" in old_key:
                    new_key = old_key.replace("stage1", "1")
                    para[new_key] = para.pop(old_key)
                elif "stage2" in old_key:
                    new_key = old_key.replace("stage2", "2")
                    para[new_key] = para.pop(old_key)
                elif "stage3" in old_key:
                    new_key = old_key.replace("stage3", "3")
                    para[new_key] = para.pop(old_key)
                elif "stage4" in old_key:
                    new_key = old_key.replace("stage4", "4")
                    para[new_key] = para.pop(old_key)
            model.load_state_dict(para)
    else:
        model = ELSE_MODEL_DICT[name](pretrained=False)
        if num_classes!=1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        if pretrained:
            checkpoint = torch.load('pretrained/'+dataset+'_'+name+'.pth',map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])       
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False, split=['A', 'B', 'C', 'D']):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='imagenet' or name=='imagenet-0.5':
        num_classes=1000
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        train_dst = datasets.ImageNet(data_root, split='train', transform=train_transform)
        val_dst = datasets.ImageNet(data_root, split='val', transform=val_transform)
    elif name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        train_dst = datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name=='svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst


    
