#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/05/31 10:12
# @Author  : Tao Chen
# @File    : sweep_optuna.py

import os
import sys
import yaml
import shutil
import optuna
import argparse

from main import main as _main
from main import get_args_parser
from functools import partial
from typing import Optional, List, Any
from copy import deepcopy


def default_args(args):
    default_args_dict = {'eval': False, 'resume': None, 'tags': None, 'variant_file': None, 'options': None,
                         'force_override': False}

    if args.parallel_mode == 'ddp':
        raise ValueError('ddp is not supported yet')

    for k, v in default_args_dict.items():
        setattr(args, k, v)


def objective(trial, args, space):
    print(f"Trial {trial.number} is starting.\n")
    with open(args.sweep_log, 'a') as f:
        f.write(f"Trial {trial.number} is starting.")
    _args = deepcopy(args)
    _args.name = '[sweep_optuna]{}/{}'.format(_args.name, trial.number)
    _args.group = '[sweep_optuna]{}'.format(_args.name)
    default_args(_args)
    for k, v in space.items():
        setattr(_args, k, SweepConfig.suggest_handle(trial, v['type'])(k, **v['meta']))

    metric = _main(_args)

    with open(_args.sweep_log, 'a') as f:
        info = 'Trial value: {}'.format(metric) + '\n' + 'Trial hyperparameters: {}'.format(trial.params) + '\n'
        f.write(info)

    return metric


def callback(study, trial, **kwargs):
    # 每次试验结束后被调用
    best_trial = study.best_trial
    with open(kwargs['sweep_log'], 'a') as f:
        f.write('Current best is trial {} with value: {}\n\n'.format(best_trial.number, best_trial.value))


class SweepConfig(object):
    def __init__(self, space: dict, sampler: str = 'TPE', count: int = None, **kwargs):
        self.count = count
        self.space = space
        self.sampler_name = sampler
        if sampler.lower() == 'tpe':
            self.sampler = optuna.samplers.TPESampler(**kwargs)
        elif sampler.lower() == 'random':
            self.sampler = optuna.samplers.RandomSampler(**kwargs)
        elif sampler.lower() == 'cmaes':
            self.sampler = optuna.samplers.CmaEsSampler(**kwargs)
        elif sampler.lower() == 'grid':
            param_grid = {}
            self.count = 1
            for k, v in space.items():
                assert v['type'] == 'categorical'
                param_grid[k] = v['meta']['choices']
                self.count *= len(v['meta']['choices'])
            self.sampler = optuna.samplers.GridSampler(param_grid, **kwargs)
        else:
            raise ValueError('Unsupported sampler: {}'.format(sampler))

    def __str__(self):
        _dict = {'sampler': self.sampler_name,
                 'space': self.space,
                 'count': self.count, }

        return yaml.dump(_dict)[:-1]

    @staticmethod
    def suggest_handle(trial, suggest_type):
        if suggest_type == 'float':
            return trial.suggest_float
        elif suggest_type == 'int':
            return trial.suggest_int
        elif suggest_type == 'categorical':
            return trial.suggest_categorical
        else:
            raise ValueError('Unsupported suggest type: {}'.format(suggest_type))

    @staticmethod
    def float(low: float, high: float, step: Optional[float] = None, log: bool = False):
        return {'type': 'float', 'meta': {'low': low, 'high': high, 'step': step, 'log': log}}

    @staticmethod
    def int(low: int, high: int, step: int = 1, log: bool = False):
        return {'type': 'int', 'meta': {'low': low, 'high': high, 'step': step, 'log': log}}

    @staticmethod
    def categorical(choices: List[Any]):
        return {'type': 'categorical', 'meta': {'choices': choices}}


def sweep(sweep_config):
    from build_config import config

    config()
    parser = argparse.ArgumentParser('Model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    default_args(args)

    if os.path.exists('./runs/[sweep_optuna]{}'.format(args.name)):
        q = input('\033[1;31mA sweep with the same name "{}" has already existed, '
                  'whether to override [y/n]: \033[0m'.format(args.name))
        if q == 'y' or q == 'Y':
            shutil.rmtree('./runs/[sweep_optuna]{}'.format(args.name))
        else:
            sys.exit()

    sweep_dir = './runs/[sweep_optuna]{}'.format(args.name)
    os.makedirs(sweep_dir, exist_ok=True)
    shutil.copy("./config/config.yml", sweep_dir)
    args.config_file = sweep_dir + '/config.yml'
    args.sweep_log = sweep_dir + '/sweep_log.txt'

    print('Full sweep configuration:\n' + str(sweep_config))
    with open(os.path.join(sweep_dir, 'sweep_config.yml'), "w") as f:
        f.write(str(sweep_config))
    print("Full sweep configuration saved to '{}'".format(os.path.join(sweep_dir, 'sweep_config.yml')))

    study = optuna.create_study(study_name=args.name, sampler=sweep_config.sampler,
                                direction="maximize" if args.better == 'large' else "minimize")
    study.optimize(partial(objective, args=args, space=sweep_config.space), n_trials=sweep_config.count,
                   callbacks=[partial(callback, sweep_log=args.sweep_log)])

    best_trial = study.best_trial
    with open(args.sweep_log, 'a') as f:
        f.write('Best is trial {} with value: {}'.format(best_trial.number, best_trial.value) + '\n')
        f.write("Best hyperparameters is: {}".format(study.best_params) + '\n')


if __name__ == '__main__':
    sweep_config = SweepConfig(
        space={
            'lr': SweepConfig.categorical([1, 2, 3])
        },
        count=10,
        sampler='grid'
    )
    sweep(sweep_config)
