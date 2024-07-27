# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/4/8 16:17 
# ide： PyCharm
import torch

# used in main
tb_args = dict(
    train_vs_val_metric=('loss', 'acc'),
    train_metric=('loss',),
)


def add_train_vs_eval_metric(tensorboard_writer, train_stats, val_stats, epoch, flush=False):
    for metric in tb_args['train_vs_val_metric']:
        tensorboard_writer.add_scalars('Metric/{}'.format(metric),
                                       {'Train_{}'.format(metric): train_stats[metric],
                                        'Val_{}'.format(metric): val_stats[metric]},
                                       epoch)
    if flush:
        tensorboard_writer.flush()


def add_train_metric(tensorboard_writer, train_stats, epoch, flush=False):
    for metric in tb_args['train_metric']:
        tensorboard_writer.add_scalars('Metric/{}'.format(metric),
                                       {'Train_{}'.format(metric): train_stats[metric]},
                                       epoch)
    if flush:
        tensorboard_writer.flush()


# torch>=1.3
def add_net_graph(tensorboard_writer, model, dataset):
    samlpe = dataset[0]
    dummy_input = []
    for i in range(len(samlpe)):
        if type(samlpe[i]) == torch.Tensor:
            dummy_input.append(samlpe[i].unsqueeze(dim=0))
    tensorboard_writer.add_graph(model, *dummy_input)
