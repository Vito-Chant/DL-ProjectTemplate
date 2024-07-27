# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 19:28 
# ide： PyCharm

from utils import registry
from .rafdb import Dataset as rafdbset

DATA_BUILD_FUNCS = registry.Registry('dataset build functions')


@DATA_BUILD_FUNCS.register_with_name(module_name='rafdb')
def build_rafdb(resize=224, **kwargs):
    return rafdbset(mode=kwargs['mode'], resize=resize)
