# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/7/14 15:32 
# ide： PyCharm

import torch
import matplotlib.pyplot as plt

from utils import metric


class TrainEnginePlugin():
    def __init__(self, args, tensorboard_writer, wandb_run, disable=False):
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb_run
        self.args = args
        self.disable = disable

    def pre_process(self, **kwargs):
        if not self.disable:
            pass

    def process(self, **kwargs):
        if not self.disable:
            pass

    def post_process(self, **kwargs):
        if not self.disable:
            pass


class EvalEnginePlugin():
    def __init__(self, args, tensorboard_writer=None, wandb_run=None, disable=False):
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb_run
        self.args = args
        self.disable = disable

        self.tb_args = {'val_embedding_freq': 5,
                        'val_embedding_layer': None,  # set None to forbid log embedding
                        }
        self.wandb_args = {}

        self.embedding = None
        self.metadata = None
        self.handle = None
        self.confusion_matrix = metric.ConfusionMatrix(7)

    def pre_process(self, **kwargs):
        if not self.disable:
            if self.tensorboard_writer is not None:
                if hasattr(kwargs['model'], 'module'):
                    model_without_module = kwargs['model'].module
                else:
                    model_without_module = kwargs['model']
                if type(self.tb_args['val_embedding_freq']) is int and self.tb_args[
                    'val_embedding_freq'] > 0 and kwargs['epoch'] % self.tb_args['val_embedding_freq'] == 0 and \
                        self.tb_args['val_embedding_layer'] is not None:
                    self.embedding = []
                    self.metadata = []
                    if self.tb_args['val_embedding_layer'] in list(dict(model_without_module.named_modules()).keys()):
                        def get_embedding(module, input, output):
                            if len(output.shape) == 2:
                                self.embedding.append(output.cpu().detach())

                        layer = dict(model_without_module.named_modules())[self.tb_args['val_embedding_layer']]
                        self.handle = layer.register_forward_hook(get_embedding)

    def process(self, **kwargs):
        if not self.disable:
            if self.args.model == None:
                pass
            else:  # default
                self.confusion_matrix.update(kwargs['outputs'], kwargs['targets'])

                if self.embedding is not None and self.handle is None:
                    self.embedding.append(kwargs['outputs'].cpu().detach())
                if self.metadata is not None:
                    self.metadata.append(kwargs['targets'].cpu().detach())

    def post_process(self, **kwargs):
        if not self.disable:
            if self.tensorboard_writer is not None:
                if self.embedding is not None:
                    embedding = torch.cat(self.embedding, dim=0)
                    metadata = torch.cat(self.metadata)
                    self.tensorboard_writer.add_embedding(mat=embedding, metadata=metadata, tag='Val Embedding',
                                                          global_step=kwargs['epoch'])
                    if self.handle is not None:
                        self.handle.remove()

                cm = self.confusion_matrix.plot()
                self.tensorboard_writer.add_figure(tag='Confusion matrix', figure=cm, global_step=kwargs['epoch'])

            if self.wandb_run is not None:
                # cm = wandb.plot.confusion_matrix(y_true=self.confusion_matrix.labels, preds=self.confusion_matrix.preds)
                # self.wandb_run.log({"Confusion matrix": cm})
                cm = self.confusion_matrix.plot()
                self.wandb_run.log({"Confusion matrix": cm})
                plt.close(cm)
