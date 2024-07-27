# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 18:19 
# ide： PyCharm
# modified from mmcv

import inspect

from functools import partial


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        self._init_kwargs_dict = dict()
        self._artifact_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    def __len__(self):
        return len(self._module_dict)

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def init_kwargs_dict(self):
        return self._init_kwargs_dict

    @property
    def artifact(self):
        return self._artifact_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_with_name(self, module_name=None, artifact=None, force=False):
        return partial(self.register, module_name=module_name, artifact=artifact, force=force)

    def register(self, module_build_function, module_name=None, force=False):
        """Register a module build function.
        Args:
            artifact : a list of dict, each dict must and only can have three key: 'name', 'type' and 'obj'.
        """
        if not inspect.isfunction(module_build_function):
            raise TypeError('module_build_function must be a function, but got {}'.format(
                type(module_build_function)))

        args = inspect.getfullargspec(module_build_function).args
        defaults = inspect.getfullargspec(module_build_function).defaults
        init_kwargs = {}
        if defaults is None:
            defaults = ()
        num_undefaults = len(args) - len(defaults)
        for i, k in enumerate(args):
            if i < num_undefaults:
                init_kwargs[k] = None
            else:
                init_kwargs[k] = defaults[i - num_undefaults]
        if 'optim' in self._name:
            if 'max_norm' not in init_kwargs: init_kwargs['max_norm'] = None
            if 'no_weight_decay_on_bn' not in init_kwargs: init_kwargs['no_weight_decay_on_bn'] = False
            if 'no_weight_decay_on_bias' not in init_kwargs: init_kwargs['no_weight_decay_on_bias'] = False

        if module_name is None:
            module_name = module_build_function.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))

        self._module_dict[module_name] = module_build_function
        self._init_kwargs_dict[module_name] = init_kwargs

        return module_build_function
