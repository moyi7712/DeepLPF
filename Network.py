from Models import DeepLPF
import tensorflow as tf
import tensorflow.keras as keras


class Net(object):
    def __init__(self, config):
        self.lamda_l1 = config.l1
        self.lamda_ms = config.ms
        self.model = DeepLPF()
        self.optimizer = keras.optimizers.Adam(lr=1e-4,epsilon=1e-8)

    def rgb2Lab(self, image):
        image = tf.where(image > 0.04045, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)
        X = image[:, :, 0] * 0.412453 + image[:, :, 1] * 0.357580 + image[:, :, 2] * 0.180423
        Y = image[:, :, 0] * 0.212671 + image[:, :, 1] * 0.715160 + image[:, :, 2] * 0.072169
        Z = image[:, :, 0] * 0.019334 + image[:, :, 1] * 0.119193 + image[:, :, 2] * 0.950227
        eps = 6 / 29
        Xn = (X / 0.950456)
        Yn = (Y / 1.0)
        Zn = (Z / 1.088754)
        f = lambda value: tf.where(value > eps ** 3, value ** (1 / 3), ((1 / 3) * (1 / eps) ** 2) * value + 4 / 29)
        L = 116 * f(Yn) - 16
        a = 500 * (f(Xn) - f(Yn))
        b = 200 * (f(Yn) - f(Zn))
        Lab = tf.concat(values=[L, a, b], axis=3)
        return Lab

    def loss(self, predict, label):
        predict_Lab = self.rgb2Lab(predict)
        label_Lab = self.rgb2Lab(label)
        loss_L = tf.abs(predict_Lab - label_Lab)
        loss_ms_ssim = tf.image.ssim_multiscale(predict_Lab[:, :, 0], label_Lab[:, :, 0], 100.0,
                                                filter_size=5, filter_sigma=0.5)
        loss = self.lamda_l1*loss_L + self.lamda_ms*loss_ms_ssim
        return loss

    def train(self, inputs, labels):
        with tf.GradientTape() as tape:
            predict = self.model(inputs)
            loss = self.loss(predict, labels)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
