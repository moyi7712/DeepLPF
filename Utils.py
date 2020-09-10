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


def rgb2Lab(image):
    image = tf.where(image > 0.04045, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)
    X = image[:, :, :, 0] * 0.412453 + image[:, :, :, 1] * 0.357580 + image[:, :, :, 2] * 0.180423
    Y = image[:, :, :, 0] * 0.212671 + image[:, :, :, 1] * 0.715160 + image[:, :, :, 2] * 0.072169
    Z = image[:, :, :, 0] * 0.019334 + image[:, :, :, 1] * 0.119193 + image[:, :, :, 2] * 0.950227
    eps = 6 / 29
    Xn = (X / 0.950456)
    Yn = (Y / 1.0)
    Zn = (Z / 1.088754)
    f = lambda value: tf.where(value > eps ** 3, value ** (1 / 3), ((1 / 3) * (1 / eps) ** 2) * value + 4 / 29)
    L = tf.expand_dims(116 * f(Yn) - 16, axis=3)
    a = tf.expand_dims(500 * (f(Xn) - f(Yn)), axis=3)
    b = tf.expand_dims(200 * (f(Yn) - f(Zn)), axis=3)
    L = L / 100
    a = ((a / 110) - 1) / 2
    b = ((b / 110) - 1) / 2
    Lab = tf.concat(values=[L, a, b], axis=3)
    return Lab
