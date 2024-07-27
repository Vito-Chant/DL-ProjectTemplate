# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 18:22 
# ide： PyCharm

import torch
from utils import registry
from .resnet import *
from .vit import ViT

MODEL_BUILD_FUNCS = registry.Registry('model and criterion build functions')
'''
The return value of a model build function must be one of two ways:
    (1) return model, criterion
    (2) return model, (criterion_train, criterion_eval)
For way (2), if criterion_eval is assigned to "None", this run will not execute “evaluate”
'''


@MODEL_BUILD_FUNCS.register_with_name(module_name='resnet')
def build_resnet(arch='resnet18', num_classes=7, img_channels=3, pretrained=False):
    model_handle = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d,
        'wide_resnet50_2': wide_resnet50_2,
        'wide_resnet101_2': wide_resnet101_2
    }
    model = model_handle[arch](pretrained, num_classes=num_classes, img_channels=img_channels)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion_train = loss_fn_1
    # criterion_eval = loss_fn_2

    return model, criterion
    # return model, (criterion_train, criterion_eval)


@MODEL_BUILD_FUNCS.register_with_name(module_name='vit')
def build_vit(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64,
              dropout=0., emb_dropout=0.):
    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth,
                heads=heads, mlp_dim=mlp_dim, pool=pool, channels=channels, dim_head=dim_head, dropout=dropout,
                emb_dropout=emb_dropout)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion
