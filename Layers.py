import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from Utils import PaddingMode


class Padding(keras.layers.Layer):
    def __init__(self, size=1, mode=PaddingMode.zeros):
        super(Padding, self).__init__()
        self.size = size
        self.mode = mode.value

    def call(self, inputs, **kwargs):
        outputs = tf.pad(inputs,
                         [[0, 0],
                          [self.size, self.size],
                          [self.size, self.size],
                          [0, 0]],
                         mode=self.mode)
        return outputs


class Conv(keras.layers.Layer):
    def __init__(self, channels, name, activation='leaky_relu',
                 is_pooling=True, padding_mode=PaddingMode.reflect, norm=None):
        super(Conv, self).__init__(channels, name)
        self.stride = 1
        self.bias = True
        self.kernel_size = 3
        self.padding_size = int(self.kernel_size / 2)
        self.is_pooling = is_pooling
        self.pad = Padding(size=self.padding_size, mode=padding_mode)
        self.activation_func = getattr(tf.nn, activation) if activation else None
        self.conv = keras.layers.Conv2D(filters=channels,
                                        kernel_size=self.kernel_size,
                                        strides=self.stride,
                                        use_bias=self.bias,
                                        activation=self.activation_func,
                                        name=name + '_Conv_0')

        self.pool = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.is_norm = True
        if norm == 'bn':
            self.norm = keras.layers.BatchNormalization()
        elif norm == 'in':
            self.norm = tfa.layers.InstanceNormalization()
        else:
            self.is_norm = False

    def call(self, inputs, **kwargs):
        outputs = self.conv(self.pad(inputs))
        if self.is_norm:
            outputs = self.norm(outputs)
        if self.is_pooling:
            outputs = self.pool(outputs)

        return outputs


class LocalConv(Conv):
    def __init__(self, channels, name, activation='leaky_relu',
                 is_pooling=True, padding_mode=PaddingMode.reflect, norm=None):
        super(LocalConv, self).__init__(channels=channels, name=name,
                                        activation=activation,
                                        is_pooling=is_pooling,
                                        padding_mode=padding_mode,
                                        norm=norm)
        self.conv_1 = keras.layers.Conv2D(filters=channels,
                                          kernel_size=self.kernel_size,
                                          strides=self.stride,
                                          use_bias=self.bias,
                                          activation=self.activation_func,
                                          name=name + '_Conv_1')

    def call(self, inputs, **kwargs):
        outputs = self.conv(self.pad(inputs))
        outputs = self.conv_1(self.pad(outputs))
        if self.is_norm:
            outputs = self.norm(outputs)
        if self.is_pooling:
            pooling = self.pool(outputs)
        else:
            pooling = None
        return outputs, pooling


class Upsample(keras.layers.Layer):
    def __init__(self, channels, scale, name, mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
        super(Upsample, self).__init__()
        self.scale = scale
        self.mode = mode

        self.conv = keras.layers.Conv2D(filters=channels, kernel_size=1, strides=1, name=name + '_Conv')

    def call(self, inputs, **kwargs):
        _, w, h, _ = tf.shape(inputs)
        w = w * self.scale
        h = h * self.scale
        output = tf.image.resize(inputs, (w, h), method=self.mode)
        output = self.conv(output)
        return output
