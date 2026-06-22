import os
import yaml
from typing import Any, Dict, Union
import torch.nn as nn


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=1)
    elif activation == 'rrelu':
        return nn.RReLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'mish':
        return nn.Mish()
    elif activation == 'glu':
        return nn.GLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lin':
        return nn.Identity()
    else:
        return None


def check_dir(path):
    """
    :param path(str): the path need to check whether exists in or not
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_config(config: Union[str, Dict], inheritance_key: str = 'INHERIT') -> Dict[str, Any]:
    """Reads YAML configuration file with nested inheritance from other YAML files.
    Arguments:
        config {Union[str, Dict]} -- Configuration path/dictionary
    Keyword Arguments:
        inheritance_key {str} -- String used for inheritance paths (default: {'FROM'})
    Returns:
        Dict[str, Any] -- Configuration dictionary
    """
    if isinstance(config, str):
        config_dict = yaml.safe_load(open(config))
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f'Expected config to be a str or dict but got {type(config)}.')

    if inheritance_key in config_dict:
        for yaml_file in config_dict[inheritance_key]:
            parent_config = load_config(yaml_file, inheritance_key)
            parent_config.update(config_dict)
            config_dict = parent_config

    return config_dict


def change_keys(ckpt, old_name=None, new_name='', mode='replace'):
    """remove the string in checkpoints keys
    :params ckpt(str): checkpoints
    :params name(str): to remove string
    """
    for key in list(ckpt.keys()):
        if mode == 'replace':
            if old_name in key:
                ckpt[key.replace(old_name, new_name)] = ckpt[key]
        elif mode == 'add':
            ckpt[new_name + key] = ckpt[key]
        else:
            raise ValueError(f'No support {mode} mode!')

        del ckpt[key]

    return ckpt


def str2list(strings):
    if isinstance(strings, str):
        strings = strings.replace(' ', ',')
        strings = strings.split(',') if ',' in strings else [strings]
    elif isinstance(strings, list):
        return strings
    else:
        raise TypeError(f'No support {type(strings)} type!')
    return strings

