# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 21:16 
# ide： PyCharm

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from config import abs_path


def rafdb_transform(mode, resize=224):
    if mode == 'train':
        return torchvision.transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.57535914, 0.44928582, 0.40079932],
                                 std=[0.20735591, 0.18981615, 0.18132027]),
            transforms.RandomErasing(scale=(0.02, 0.25)), ])
    elif mode == 'eval':
        return torchvision.transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.57535914, 0.44928582, 0.40079932],
                                 std=[0.20735591, 0.18981615, 0.18132027]), ])
    else:
        raise ValueError(f'unknown {mode}')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, transform=True, resize=224):
        super(Dataset, self).__init__()
        txt_root = {
            'train': abs_path.rafdb['train'],
            'eval': abs_path.rafdb['test']
        }
        f = open(txt_root[mode], 'r')
        imgs = []
        for line in f:
            line = line.rstrip()
            line_split = line.split(' ')
            imgs.append((line_split[0], int(line_split[1])))

        self.imgs = imgs
        self.mode = mode
        self.transform = rafdb_transform(mode=mode, resize=resize) if transform else None

    def __getitem__(self, index):
        root, label_expression = self.imgs[index]
        img = Image.open(root)

        if self.transform:
            transform = self.transform(self.mode)
            img = transform(img)

        return img, label_expression

    def __len__(self):
        return len(self.imgs)
