from Models import DeepLPF
import tensorflow as tf
import tensorflow.keras as keras
import os

class Net(object):
    def __init__(self, config):
        self.lamda_l1 = config.l1
        self.lamda_ms = config.ms
        self.model = DeepLPF()
        self.optimizer = keras.optimizers.Adam(lr=1e-4, epsilon=1e-8)

        self.summary_writer = tf.summary.create_file_writer(os.path.join(config.checkpoint_prefix, 'summary'))
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=config.checkpoint_prefix, max_to_keep=20)
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint)

    @staticmethod
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

    def loss(self, predict, label):
        predict_Lab = self.rgb2Lab(predict)
        label_Lab = self.rgb2Lab(label)

        L_p, _, _ = tf.split(predict_Lab, 3, axis=3)
        L_l, _, _ = tf.split(label_Lab, 3, axis=3)
        loss_L = tf.reduce_mean(tf.abs(predict_Lab - label_Lab))
        loss_ms_ssim = tf.image.ssim_multiscale(L_p, L_l, max_val=1.0,
                                                filter_size=5, filter_sigma=0.5)
        loss_ms_ssim = tf.reduce_mean(1-loss_ms_ssim)
        loss = tf.reduce_mean(self.lamda_l1*loss_L + self.lamda_ms*loss_ms_ssim)
        return loss, loss_L, loss_ms_ssim

    def train(self, inputs, labels):
        with tf.GradientTape() as tape:
            predict = self.model(inputs)
            loss, loss_L, loss_ssim = self.loss(predict, labels)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, loss_L, loss_ssim

