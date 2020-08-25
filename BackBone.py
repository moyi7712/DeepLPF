import tensorflow as tf
import tensorflow.keras as keras
from Layers import Conv, LocalConv, Upsample


class Unet(keras.models.Model):
    def __init__(self):
        super(Unet, self).__init__()

        self.down_layer_0 = LocalConv(channels=16, name='Down_0')
        self.down_layer_1 = LocalConv(channels=32, name='Down_1')
        self.down_layer_2 = LocalConv(channels=64, name='Down_2')
        self.down_layer_3 = LocalConv(channels=128, name='Down_3')
        self.down_layer_4 = LocalConv(channels=128, name='Down_4', is_pooling=False)
        self.upsample_layer_0 = Upsample(channels=128, scale=2, name='Upsample_0')
        self.upsample_layer_1 = Upsample(channels=128, scale=2, name='Upsample_1')
        self.upsample_layer_2 = Upsample(channels=64, scale=2, name='Upsample_2')
        self.upsample_layer_3 = Upsample(channels=32, scale=2, name='Upsample_3')
        self.up_layer_3 = LocalConv(channels=128, name='Up_3', is_pooling=False)
        self.up_layer_2 = LocalConv(channels=64, name='Up_2', is_pooling=False)
        self.up_layer_1 = LocalConv(channels=32, name='Up_1', is_pooling=False)
        self.up_layer_0 = LocalConv(channels=16, name='Up_0', is_pooling=False)
        self.last_layer = LocalConv(channels=3, name='Last', is_pooling=False)
        self.final_layer = Conv(channels=64, name='Final', is_pooling=False, activation=None)

    def call(self, inputs, training=None, mask=None):
        c0, x = self.down_layer_0(inputs)
        c1, x = self.down_layer_1(x)
        c2, x = self.down_layer_2(x)
        c3, x = self.down_layer_3(x)
        x, _ = self.down_layer_4(x)

        x = self.upsample_layer_0(x)
        x = tf.concat(values=[x, c3], axis=3)
        x, _ = self.up_layer_3(x)

        x = self.upsample_layer_1(x)
        x = tf.concat(values=[x, c2], axis=3)
        x, _ = self.up_layer_2(x)

        x = self.upsample_layer_2(x)
        x = tf.concat(values=[x, c1], axis=3)
        x, _ = self.up_layer_1(x)

        x = self.upsample_layer_3(x)
        x = tf.concat(values=[x, c0], axis=3)
        x, _ = self.up_layer_0(x)

        last, _ = self.last_layer(x)
        last = last+inputs
        output = self.final_layer(last)
        return output




