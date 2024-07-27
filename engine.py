# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 18:18 
# ide： PyCharm

import math
import torch

from typing import Iterable
from utils import misc, metric, engine_plugin


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: Iterable,
        epoch: int,
        args,
        lr_scheduler=None,
        logger=None,
        tensorboard_writer=None,
        wandb_run=None
):
    if args.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.update(lr=misc.get_lr(optimizer))
    header = '\033[0;34mEpoch [{}]\033[0m'.format(epoch)
    device = torch.device(args.device)

    disable = args.disable_engine_plugin == 'train' or args.disable_engine_plugin == 'all'
    plugin = engine_plugin.TrainEnginePlugin(args, tensorboard_writer, wandb_run, disable=disable)
    plugin.pre_process()  # TODO pass/modify parameters
    for metadata in metric_logger.log_every(data_loader, args.train_print_freq, header, logger=logger):
        # TODO
        if args.model == None:
            pass
        else:  # default
            samples = metadata[0].to(device)
            targets = metadata[1].to(device)
            if args.amp:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)
            acc1, _ = metric.accuracy(outputs, targets, topk=(1, 5))
            metric_logger.update(acc=acc1.item())
            plugin.process()  # TODO pass/modify parameters
        #

        if not math.isfinite(loss):
            raise ValueError("Loss is {}, stopping training".format(loss))

        if args.amp:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()

        if args.optim_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.optim_max_norm, norm_type=2)
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    plugin.post_process()  # TODO pass/modify parameters

    if lr_scheduler is not None:
        lr_scheduler.step()

    metric_logger.synchronize_between_processes()
    print(header + ':' + '  ' + "Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return resstat


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        epoch: int,
        args,
        logger=None,
        tensorboard_writer=None,
        wandb_run=None
):
    model.eval()
    criterion.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '\033[0;32mEval\033[0m'
    device = torch.device(args.device)

    disable = args.disable_engine_plugin == 'eval' or args.disable_engine_plugin == 'all'
    plugin = engine_plugin.EvalEnginePlugin(args, tensorboard_writer, wandb_run, disable=disable)
    plugin.pre_process(model=model, epoch=epoch)  # TODO pass/modify parameters
    for metadata in metric_logger.log_every(data_loader, args.eval_print_freq, header, logger=logger):
        # TODO
        if args.model == None:
            pass
        else:  # default
            samples = metadata[0].to(device)
            targets = metadata[1].to(device)
            if args.amp:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)
            acc1, _ = metric.accuracy(outputs, targets, topk=(1, 5))
            metric_logger.update(acc=acc1.item())
            plugin.process(outputs=outputs, targets=targets)  # TODO pass/modify parameters
        #
        metric_logger.update(loss=loss.item())

    plugin.post_process(epoch=epoch)  # TODO pass/modify parameters

    metric_logger.synchronize_between_processes()
    print(header + ':' + '  ' + "Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return stats
