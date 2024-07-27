# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/24 20:49 
# ide： PyCharm

import os
import yaml
from models import MODEL_BUILD_FUNCS
from optim import OPTIM_BUILD_FUNCS
from datasets import DATA_BUILD_FUNCS


def build_config():
    config = dict(
        model=MODEL_BUILD_FUNCS.init_kwargs_dict,
        optim=OPTIM_BUILD_FUNCS.init_kwargs_dict,
        dataset=DATA_BUILD_FUNCS.init_kwargs_dict
    )

    with open("./config/config.yml", "w") as f:
        f.write(yaml.dump(config))


def update_config():
    assert os.path.isfile("./config/config.yml")
    with open("./config/config.yml", 'r', encoding='utf-8') as f:
        old_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    new_config = dict(
        model=MODEL_BUILD_FUNCS.init_kwargs_dict,
        optim=OPTIM_BUILD_FUNCS.init_kwargs_dict,
        dataset=DATA_BUILD_FUNCS.init_kwargs_dict
    )

    for k in new_config.keys():
        if k in set(old_config.keys()):
            for module in new_config[k].keys():
                if module in set(old_config[k].keys()):
                    for arg in new_config[k][module].keys():
                        if arg in set(old_config[k][module].keys()):
                            new_config[k][module][arg] = old_config[k][module][arg]

    with open("./config/config.yml", "w") as f:
        f.write(yaml.dump(new_config))


def config():
    if not os.path.exists("./config/config.yml"):
        build_config()
    else:
        update_config()


if __name__ == '__main__':
    config()
