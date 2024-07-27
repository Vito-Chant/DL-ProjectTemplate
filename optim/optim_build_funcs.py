# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 19:27 
# ide： PyCharm

import torch
from utils import registry

OPTIM_BUILD_FUNCS = registry.Registry('optimizer and scheduler build functions')
'''
There are three additional hyperparameters that control the optimizer behavior:
    max_norm: Used to trim the gradient so that it does not exceed the specified value, 
            the appropriate value is conducive to the stability of the training, but default = None
    no_weight_decay_on_bn: Usually useful for model's performance, but default = False
    no_weight_decay_on_bias: Usually useful for model's performance, but default = False
'''


@OPTIM_BUILD_FUNCS.register_with_name(module_name='sgd_multistep')
def build_sgd_multistep(lr=0.01, momentum=0, weight_decay=0, milestones=[60, 100], gama=0.1, **kwargs):
    optimizer = torch.optim.SGD(kwargs['params'], lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gama)

    return optimizer, lr_scheduler


@OPTIM_BUILD_FUNCS.register_with_name(module_name='adam_multistep')
def build_adam_multistep(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, milestones=[5, 10],
                         gama=0.1, **kwargs):
    optimizer = torch.optim.Adam(kwargs['params'], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                 amsgrad=amsgrad)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gama)

    return optimizer, lr_scheduler


@OPTIM_BUILD_FUNCS.register_with_name(module_name='sgd_step')
def build_sgd_step(lr=0.005, momentum=0.9, weight_decay=0.0005, step_size=30, gamma=0.5, **kwargs):
    optimizer = torch.optim.SGD(kwargs['params'], lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return optimizer, lr_scheduler


@OPTIM_BUILD_FUNCS.register_with_name(module_name='adam_step')
def build_adam_step(lr=0.005, step_size=30, gamma=0.5, **kwargs):
    optimizer = torch.optim.Adam(kwargs['params'], lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return optimizer, lr_scheduler


@OPTIM_BUILD_FUNCS.register_with_name(module_name='sgd_ca')
def build_sgd_ca(lr=0.01, momentum=0, weight_decay=0, T_max=None, **kwargs):
    optimizer = torch.optim.SGD(kwargs['params'], lr=lr, momentum=momentum, weight_decay=weight_decay)
    if T_max is None:
        T_max = kwargs['args'].epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    return optimizer, lr_scheduler


@OPTIM_BUILD_FUNCS.register_with_name(module_name='adam_ca')
def build_adam_ca(lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, T_max=None, **kwargs):
    optimizer = torch.optim.Adam(kwargs['params'], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                 amsgrad=amsgrad)
    if T_max is None:
        T_max = kwargs['args'].epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    return optimizer, lr_scheduler
