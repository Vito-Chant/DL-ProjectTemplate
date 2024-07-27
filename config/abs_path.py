# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/4/4 8:20 
# ide： PyCharm

import platform

if platform.system() == 'Linux':
    rafdb = {
        'train': '/home/developers/chentao/datasets/RAF-DB/dataset_train_ct.txt',
        'test': '/home/developers/chentao/datasets/RAF-DB/dataset_test_ct.txt',
    }

    # rafdb = {
    #     'train': '/home/developers/tengjianing/zouwei/dataset/RAF-DB/dataset_train.txt',
    #     'test': '/home/developers/tengjianing/zouwei/dataset/RAF-DB/dataset_test.txt'
    # }
elif platform.system() == 'Windows':
    rafdb = {
        'train': 'H:/chent2/datasets/RAF-DB/dataset_train_ct.txt',
        'test': 'H:/chent2/datasets/RAF-DB/dataset_test_ct.txt',
    }
