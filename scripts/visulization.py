# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/4/8 20:10 
# ide： PyCharm

# # FIXME
# # Use tensorboard to visualize network structure
#
# import yaml
# import os
# import shutil
#
# from utils.tensorboard import add_net_graph
# from models import MODEL_BUILD_FUNCS
# from datasets import DATA_BUILD_FUNCS
# from torch.utils.tensorboard import SummaryWriter
#
# model_name = 'resnet'
# dataset_name = 'rafdb'
#
# with open("../config/config.yml", 'r', encoding='utf-8') as f:
#     config = yaml.load(f.read(), Loader=yaml.FullLoader)
#
# model = MODEL_BUILD_FUNCS.get('resnet')(**config['model'][model_name])
# dataset = DATA_BUILD_FUNCS.get('rafdb')(**config['dataset'][dataset_name], mode='eval')
#
# output_path = '../analysis/net_graph/{}'.format(model_name)
# if os.path.exists(output_path):
#     shutil.rmtree(output_path)
# writer = SummaryWriter(log_dir='./runs/test/tensorboard')
# add_net_graph(writer, model, dataset)
# writer.close()
