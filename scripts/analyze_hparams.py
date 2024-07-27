# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/4/11 17:21 
# ide： PyCharm
#
# used for tensorboard

import os
import yaml
import torch
import shutil
from torch.utils.tensorboard import SummaryWriter


def get_best_metric(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    best_metric_holder = checkpoint['best_metric']
    return {'hparam/' + best_metric_holder.metric_name: best_metric_holder.best_res}


def resolve_hparam(hparam_dict):
    def _check(v):
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, str) \
                or isinstance(v, bool) or isinstance(v, torch.Tensor):
            return v
        else:
            return str(v)

    new_dict = {}
    for k, v in hparam_dict.items():
        new_dict[k] = _check(v)

    return new_dict


def analyze_hyparams(writer):
    runs_list = os.listdir('../runs')

    for run in runs_list:
        variant = '../runs/{}/variant.yml'.format(run)
        checkpoint_best = '../runs/{}/checkpoint/checkpoint_best.pth'.format(run)
        if os.path.exists(variant) and os.path.exists(checkpoint_best):
            with open(variant, 'r', encoding='utf-8') as f:
                hyparams = yaml.load(f.read(), Loader=yaml.FullLoader)
            metric = get_best_metric(checkpoint_best)

            writer.add_hparams(resolve_hparam(hyparams), metric, run_name=run)
        else:
            pass


if __name__ == '__main__':
    output_path = '../analysis/hparams'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    writer = SummaryWriter(log_dir=output_path)
    analyze_hyparams(writer)
    writer.close()
