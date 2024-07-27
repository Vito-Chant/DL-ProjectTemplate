# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/26 21:50 
# ide： PyCharm

import torch
import os

from tqdm import tqdm
from models.resnet import resnet18
from datasets.rafdb import Dataset as rafdb
from torch.utils.data import DataLoader

# device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')

# model
model = resnet18()
model.to(device)
state_dict = torch.load(your_pretrained_model)
if hasattr(model, 'module'):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)
model.eval()

# data
dataset = rafdb(mode='eval')
dataloader = DataLoader(dataset, 32, num_workers=4)

with torch.no_grad():
    for samples, targets in tqdm(dataloader):
        samples = samples.to(device)
        outputs = model(samples)
        # TODO something else
