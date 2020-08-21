import numpy as np
import tensorflow as tf


def sign01(value):
    return 0.5 * (tf.sign(value) + 1)


def tanh01(value):
    return 0.5 * (tf.tanh(value) + 1)

