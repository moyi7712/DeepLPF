import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from enum import Enum
import yaml


def sign01(value):
    return 0.5 * (tf.sign(value) + 1)


def tanh01(value):
    return 0.5 * (tf.tanh(value) + 1)



class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

    def HyperPara(self):
        return dict_to_object(self._config['HyperPara'])

    def Dataset(self):
        return dict_to_object(self._config['Dataset'])


class resize_method(Enum):
    INTER_AREA = 3
    INTER_BITS = 5
    INTER_BITS2 = 10
    INTER_CUBIC = 2
    INTER_LANCZOS4 = 4
    INTER_LINEAR = 1
    INTER_LINEAR_EXACT = 5
    INTER_MAX = 7
    INTER_NEAREST = 0
    INTER_TAB_SIZE = 32
    INTER_TAB_SIZE2 = 1024


class PaddingMode(Enum):
    zeros = 'CONSTANT'
    reflect = 'REFLECT'
    symmetric = 'SYMMETRIC'

