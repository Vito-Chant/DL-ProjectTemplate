# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/26 21:47
# ide： PyCharm

import os
import sys
import time
import yaml
import math
import shutil
import argparse
import datetime
import torch
import wandb
import platform

import torch.distributed as dist
import torch.multiprocessing as mp

from functools import partial
from models import MODEL_BUILD_FUNCS
from optim import OPTIM_BUILD_FUNCS
from datasets import DATA_BUILD_FUNCS
from torch.utils.data import DataLoader, DistributedSampler
from utils import misc
from utils.metric import BestMetric
from utils.logger import setup_logger
from utils.tensorboard import add_train_vs_eval_metric, add_train_metric
from torch.utils.tensorboard import SummaryWriter
from engine import evaluate, train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-n', '--name', default='test', type=str, help='the global name of this experiment')
    parser.add_argument('--project', default=None, type=str, help='the project of this experiment')
    parser.add_argument('--group', default=None, type=str, help='the group of this experiment')
    parser.add_argument('--tags', default=None, type=str, nargs='+', help='the tags of this experiment')
    parser.add_argument('--notes', default=None, type=str, help='add some notes to the experiment')

    parser.add_argument('--variant_file', default=None, type=str,
                        help='variant file path, override the variant args below')
    # variant args
    parser.add_argument('-m', '--model', default=None, type=str, help='choose model and criterion from config.yml')
    parser.add_argument('-o', '--optim', default=None, type=str,
                        help='choose optimizer and lr_scheduler from config.yml')
    parser.add_argument('-d', '--dataset', default=None, type=str, help='choose dataset from config.yml')
    parser.add_argument('-e', '--epochs', default=2, type=int, help='total training epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')  # exclude variant
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)  # exclude variant
    parser.add_argument('-s', '--seed', default=None, type=int)
    parser.add_argument('-a', '--amp', action='store_true', help="train with mixed precision, torch>=1.6")

    parser.add_argument('--options', nargs='+', action=misc.DictAction,
                        help='override some settings in the used variant file or config file, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')

    # metric for determining the best model
    parser.add_argument('--metric', default='acc', type=str,
                        help='metric used to determine whether the model is the best, '
                             'make sure that this metric is in the "stats" returned by engine.evaluate()')
    parser.add_argument('--better', default='large', type=str, choices=['large', 'small'], help='used by metric')

    # device and data parallel setting
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'],
                        help='device to use for training/testing')
    parser.add_argument('-g', '--gpu_id', default=None, type=int, nargs='+', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('-p', '--parallel_mode', default=None, type=str, choices=['dp', 'ddp'])
    # for ddp
    parser.add_argument('--nodes', default=1, type=int, metavar='N', help='used for ddp')
    parser.add_argument('--gpus', default=1, type=int, help='used for ddp, number of gpus per node')
    parser.add_argument('--node_rank', default=0, type=int, help='used for ddp, ranking within the nodes')
    parser.add_argument('--init_method', default='env://', type=str)
    parser.add_argument('--find_unused_params', action='store_true', help='used for ddp')

    # checkpoint
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--save_checkpoint_interval', default=1, type=int)
    parser.add_argument('--reserve_last_checkpoint', action='store_false')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--disable_tensorboard', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true', help='wandb offline mode')
    parser.add_argument('--save_log', action='store_true', help='save log to info.txt')
    parser.add_argument('--print_freq', default=0, type=int)
    parser.add_argument('-f', '--force_override', action='store_true',
                        help='if there is an experiment with the same name, it will be forcibly overridden')
    parser.add_argument('--disable_engine_plugin', default=None, type=str, choices=['train', 'eval', 'all'])
    parser.add_argument('--config_file', default='./config/config.yml', type=str, help='config file path')

    return parser


def main(args):
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if args.variant_file:
        assert os.path.exists(args.variant_file)
    else:
        assert args.model in MODEL_BUILD_FUNCS.module_dict
        assert args.optim in OPTIM_BUILD_FUNCS.module_dict
        assert args.dataset in DATA_BUILD_FUNCS.module_dict

    args.output_dir = './runs/{}'.format(args.name)
    args.checkpoint_output_dir = args.output_dir + '/checkpoint'
    args.tensorboard_output_dir = None if args.disable_tensorboard else args.output_dir + '/tensorboard'
    args.wandb_output_dir = None if args.disable_wandb else args.output_dir + '/wandb'

    args.resume = misc.resolve_resume(args.resume)
    if args.resume is None or args.resume is False:
        if os.path.exists(args.output_dir):
            if args.force_override:
                print('\033[1;31mThe experiment with the same name "{}" has been overriden!\033[0m'.format(args.name))
                shutil.rmtree(args.output_dir)
            else:
                q = input('\033[1;31mAn experiment with the same name "{}" has already existed, '
                          'whether to override [y/n]: \033[0m'.format(args.name))
                if q == 'y' or q == 'Y':
                    shutil.rmtree(args.output_dir)
                else:
                    return
    else:
        if type(args.resume) is not str:
            checkpoint_list = list(filter(lambda x: 'checkpoint' in x, os.listdir(args.checkpoint_output_dir)))
            try:
                checkpoint_list.remove('checkpoint_best.pth')
            except:
                pass
            checkpoint_list = sorted(checkpoint_list)
            if len(checkpoint_list) > 0:
                args.resume = os.path.join(args.checkpoint_output_dir, checkpoint_list[-1])
            else:
                args.resume = None
        else:
            assert os.path.exists(args.resume)

    if args.variant_file:
        with open(args.variant_file, 'r', encoding='utf-8') as f:
            args_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        for k, v in args_dict.items():
            setattr(args, k, v)
    else:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        for n in ('model', 'optim', 'dataset'):
            for k, v in config_dict[n][vars(args)[n]].items():
                if not hasattr(args, n + '_{}'.format(k)):
                    setattr(args, n + '_{}'.format(k), v)
    if args.options is not None:
        for k, v in args.options.items():
            setattr(args, k, v)

    if args.gpu_id is None:
        args.gpu_id = list(range(torch.cuda.device_count()))
    if args.device != 'cuda':
        args.parallel_mode = None
        args.node = 1
        args.gpus = 0
        args.node_rank = 0
        return train(0, args)
    elif args.parallel_mode is None:
        args.node = 1
        args.gpus = 1
        args.node_rank = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(lambda x: str(x), args.gpu_id))
        torch.backends.cudnn.benchmark = True
        return train(0, args)
    elif args.parallel_mode == 'dp':
        args.node = 1
        args.gpus = len(args.gpu_id)
        args.world_size = args.gpus
        args.node_rank = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(lambda x: str(x), args.gpu_id))
        torch.backends.cudnn.benchmark = True
        return train(0, args)
    elif args.parallel_mode == 'ddp':
        args.world_size = args.gpus * args.nodes
        args.batch_size_per_gpu = args.batch_size // args.world_size
        args.batch_size = args.batch_size_per_gpu * args.world_size
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        assert len(args.gpu_id) >= args.gpus
        if args.world_size < 2:
            raise ValueError("World size should be larger than 1 to use ddp")
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        raise ValueError("parallel_mode must be 'dp' or 'ddp'")


def train(local_rank, args):
    args.local_rank = local_rank
    args.rank = args.node_rank * args.gpus + local_rank
    if args.parallel_mode == 'ddp':
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(lambda x: str(x), args.gpu_id))
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size,
                                rank=args.rank)
        print('DDP initialized rank {}'.format(args.rank), flush=True)
        time.sleep(0.02)
        # torch.distributed.barrier()  # too slow
        misc.setup_for_distributed(args.rank == 0)

    os.makedirs(args.checkpoint_output_dir, exist_ok=True)
    if args.tensorboard_output_dir is not None:
        os.makedirs(args.tensorboard_output_dir, exist_ok=True)
    if not args.disable_wandb:
        os.makedirs(args.wandb_output_dir, exist_ok=True)

    root_path = os.path.dirname(__file__)
    project = root_path.split('/')[-1] if platform.system() == 'Linux' else root_path.split('\\')[-1]
    wandb_run = wandb.init(
        project=args.project if args.project is not None else project,
        config=args,
        name=args.name,
        notes=args.notes,
        group=args.group,
        tags=args.tags,
        job_type='eval' if args.eval else 'train',
        dir=args.output_dir
    ) if not args.disable_wandb and args.rank == 0 else None

    tensorboard_writer = SummaryWriter(
        log_dir=args.tensorboard_output_dir) if not args.disable_tensorboard and args.rank == 0 else None

    init_kwargs = misc.resolve_init_kwargs(args)
    model_build_func = partial(MODEL_BUILD_FUNCS.get(args.model), **init_kwargs[0])
    optim_build_func = partial(OPTIM_BUILD_FUNCS.get(args.optim), **init_kwargs[1])
    dataset_build_func = partial(DATA_BUILD_FUNCS.get(args.dataset), **init_kwargs[2])

    info_path = os.path.join(args.output_dir, 'info.txt')
    if args.rank == 0:
        if os.path.exists(info_path):
            with open(info_path, 'a') as f: f.write('\n')
    logger = setup_logger(output=info_path, distributed_rank=args.rank, color=False, name=args.name)
    logger.info("Command: " + ' '.join(sys.argv))
    logger.info('Full configuration:\n' + yaml.dump(vars(args))[:-1])
    with open(os.path.join(args.output_dir, 'args.yml'), "w") as f:
        f.write(yaml.dump(vars(args)))
    logger.info("Full configuration saved to '{}'".format(os.path.join(args.output_dir, 'args.yml')))

    if args.rank == 0:
        variant = {
            'model': args.model,
            'optim': args.optim,
            'dataset': args.dataset,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'amp': args.amp,
        }
        for k, v in vars(args).items():
            if k.startswith('model_') or k.startswith('optim_') or k.startswith('dataset_'):
                variant[k] = v
        with open(os.path.join(args.output_dir, 'variant.yml'), "w") as f:
            f.write(yaml.dump(variant))
        logger.info("Variants saved to '{}'".format(os.path.join(args.output_dir, 'variant.yml')))

    device = torch.device(args.device)

    if args.seed is not None:
        args.seed = args.seed + misc.get_rank()
    misc.init_seeds(args.seed)

    model, criterion = model_build_func()
    if type(criterion) is tuple:
        criterion_train = criterion[0]
        criterion_eval = criterion[1]
    else:
        criterion_train = criterion_eval = criterion
    if wandb_run is not None:
        wandb.save(os.path.abspath(sys.modules[model.__module__].__file__))
        criterion_script_path = os.path.abspath(sys.modules[criterion_train.__module__].__file__)
        if 'site-packages' not in criterion_script_path:
            wandb.save(criterion_script_path)
        if criterion_eval is not None:
            criterion_script_path = os.path.abspath(sys.modules[criterion_eval.__module__].__file__)
            if 'site-packages' not in criterion_script_path:
                wandb.save(criterion_script_path)

    if args.parallel_mode == 'ddp':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    criterion_train.to(device)
    if criterion_eval is not None:
        criterion_eval.to(device)

    model_without_module = model
    if args.parallel_mode == 'dp':
        model = torch.nn.DataParallel(model)
        model_without_module = model.module
    elif args.parallel_mode == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=args.find_unused_params)
        model_without_module = model.module
    else:
        pass

    if args.optim_no_weight_decay_on_bn or args.optim_no_weight_decay_on_bias:
        params_weight_decay, params_un_weight_decay = misc.separate_un_weight_decay_params(model_without_module,
                                                                                           no_batchnorm=args.optim_no_weight_decay_on_bn,
                                                                                           no_bias=args.optim_no_weight_decay_on_bias)
        optimizer, lr_scheduler = optim_build_func(
            params=[{'params': params_weight_decay}, {'params': params_un_weight_decay, 'weight_decay': 0.0}],
            args=args)
    else:
        optimizer, lr_scheduler = optim_build_func(
            params=[{'params': list(filter(lambda p: p.requires_grad, model.parameters()))}], args=args)

    dataset_train = dataset_build_func(mode='train')
    if criterion_eval is not None:
        dataset_eval = dataset_build_func(mode='eval')
    if wandb_run is not None:
        wandb.save(os.path.abspath(sys.modules[dataset_train.__module__].__file__))

    if args.parallel_mode == 'ddp':
        sampler_train = DistributedSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size_per_gpu, drop_last=True)
        if criterion_eval is not None:
            sampler_eval = DistributedSampler(dataset_eval, shuffle=False)
            batch_sampler_eval = torch.utils.data.BatchSampler(sampler_eval, args.batch_size_per_gpu, drop_last=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        if criterion_eval is not None:
            sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)
            batch_sampler_eval = torch.utils.data.BatchSampler(sampler_eval, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
                                   pin_memory=True, persistent_workers=True)
    if criterion_eval is not None:
        data_loader_eval = DataLoader(dataset_eval, batch_sampler=batch_sampler_eval, num_workers=args.num_workers,
                                      pin_memory=True, persistent_workers=True)

    if args.print_freq == 0:
        args.train_print_freq, args.eval_print_freq = 0, 0
    else:
        if criterion_eval is not None:
            assert len(data_loader_eval) > args.print_freq > 0 and args.print_freq < len(data_loader_train)
            args.train_print_freq = math.ceil(len(data_loader_train) / args.print_freq)
            args.eval_print_freq = math.ceil(len(data_loader_eval) / args.print_freq)
        else:
            assert 0 < args.print_freq < len(data_loader_train)
            args.train_print_freq = math.ceil(len(data_loader_train) / args.print_freq)

    best_metric_holder = BestMetric(metric_name=args.metric, better=args.better)

    if not (args.resume is None or args.resume is False):
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_module.load_state_dict(checkpoint['model'])
        criterion_train.load_state_dict(checkpoint['criterion_train'])
        if criterion_eval is not None:
            criterion_eval.load_state_dict(checkpoint['criterion_eval'])
        if not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_metric_holder.resume(**checkpoint['best_metric'])
            if (lr_scheduler is None) ^ ('lr_scheduler' in checkpoint):
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                logger.warning(
                    "The configuration of 'lr_scheduler'in this experiment is different from the resume checkpoint!!!")
        logger.info("Resume from checkpoint: '{}'".format(args.resume))

    # FIXME You should modify this part according to your own needs
    elif args.pretrain_model_path:
        misc.init_model_from_pretrain(model_without_module, args.pretrain_model_path, logger)

    if tensorboard_writer is not None:
        tensorboard_writer.add_text('Full configuration', yaml.dump(vars(args))[:-1], args.start_epoch)

    if args.eval and criterion_eval is not None:
        os.environ['EVAL_FLAG'] = 'TRUE'  # what is it used for?
        logger.info("Start evaluating...")
        start_time = time.time()

        eval_stats = evaluate(model, criterion_eval, data_loader_eval, epoch=-1, args=args,
                              logger=(logger if args.save_log else None), tensorboard_writer=tensorboard_writer,
                              wandb_run=wandb_run)

        log_stats = {**{f'eval/{k}': v for k, v in eval_stats.items()}}
        if misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(yaml.dump(log_stats) + "\n")

        if tensorboard_writer is not None:
            tensorboard_writer.add_text('Log', yaml.dump(log_stats))

        if wandb_run is not None:
            for k, v in log_stats.items():
                wandb.run.summary[k] = v

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('End evaluating, total time: {}'.format(total_time_str))

        if tensorboard_writer is not None:
            tensorboard_writer.close()

        if wandb_run is not None:
            wandb.finish()

        return

    logger.info("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.parallel_mode == 'ddp':
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion_train, optimizer, data_loader_train, epoch, args=args,
                                      lr_scheduler=lr_scheduler, logger=(logger if args.save_log else None),
                                      tensorboard_writer=tensorboard_writer, wandb_run=wandb_run)

        if criterion_eval is not None:
            eval_stats = evaluate(model, criterion_eval, data_loader_eval, epoch=epoch, args=args,
                                  logger=(logger if args.save_log else None), tensorboard_writer=tensorboard_writer,
                                  wandb_run=wandb_run)

        if criterion_eval is not None:
            _isbest = best_metric_holder.update(eval_stats[args.metric], epoch)
        else:
            _isbest = best_metric_holder.update(train_stats[args.metric], epoch)

        save_states = {
            'model': model_without_module.state_dict(),
            'criterion_train': criterion_train.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
            'best_metric': {'best_res': best_metric_holder.best_res, 'best_ep': best_metric_holder.best_ep}
        }
        if criterion_eval is not None:
            save_states['criterion_eval'] = criterion_eval.state_dict()
        if lr_scheduler is not None:
            save_states['lr_scheduler'] = lr_scheduler.state_dict()
        if (epoch + 1) % args.save_checkpoint_interval == 0 and args.rank == 0:
            if args.reserve_last_checkpoint:
                checkpoint_list = os.listdir(args.checkpoint_output_dir)
                try:
                    checkpoint_list.remove('checkpoint_best.pth')
                except:
                    pass
                for ckpt in checkpoint_list:
                    os.remove(os.path.join(args.checkpoint_output_dir, ckpt))
            checkpoint_path = os.path.join(args.checkpoint_output_dir, f'checkpoint_{epoch:04}.pth')
            torch.save(save_states, checkpoint_path)
        if _isbest:
            checkpoint_path = os.path.join(args.checkpoint_output_dir, 'checkpoint_best.pth')
            misc.save_on_master(save_states, checkpoint_path)

        log_stats = {**{f'train/{k}': v for k, v in train_stats.items()}}
        if criterion_eval is not None:
            log_stats.update({f'eval/{k}': v for k, v in eval_stats.items()})
        ep_paras = {'epoch': epoch}
        log_stats.update(ep_paras)
        wandb_log_dict = log_stats.copy()
        log_stats.update(best_metric_holder.summary())
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(yaml.dump(log_stats) + "\n")

        if tensorboard_writer is not None:
            tensorboard_writer.add_text('Log', yaml.dump(log_stats), epoch)
            if criterion_eval is not None:
                add_train_vs_eval_metric(tensorboard_writer, train_stats, eval_stats, epoch)
            else:
                add_train_metric(tensorboard_writer, train_stats, epoch)

        if wandb_run is not None:
            wandb_run.log(wandb_log_dict)
            for k, v in best_metric_holder.summary().items():
                wandb.run.summary[k] = v

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('End training, total time: {}'.format(total_time_str))

    if tensorboard_writer is not None:
        tensorboard_writer.close()

    if wandb_run is not None:
        wandb.alert(title='End of training', text=str(best_metric_holder), level=wandb.AlertLevel.INFO)
        wandb.finish()

    return best_metric_holder.best_res


if __name__ == '__main__':
    from build_config import config

    config()
    parser = argparse.ArgumentParser('Model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
