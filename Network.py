from Models import DeepLPF
import tensorflow as tf
import tensorflow.keras as keras
import os
import Utils


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

    def loss(self, predict, label):
        predict_Lab = Utils.rgb2Lab(predict)
        label_Lab = Utils.rgb2Lab(label)

        L_p, _, _ = tf.split(predict_Lab, 3, axis=3)
        L_l, _, _ = tf.split(label_Lab, 3, axis=3)
        loss_L = tf.reduce_mean(tf.abs(predict_Lab - label_Lab))

        loss_ms_ssim = tf.image.ssim_multiscale(L_p, L_l, max_val=1.0,
                                                filter_size=5, filter_sigma=0.5)
        loss_ms_ssim = tf.reduce_mean(1 - loss_ms_ssim)
        loss = tf.reduce_mean(self.lamda_l1 * loss_L + self.lamda_ms * loss_ms_ssim)
        return loss, loss_L, loss_ms_ssim

    def train(self, inputs, labels):
        with tf.GradientTape() as tape:
            predict = self.model(inputs)

            loss, loss_L, loss_ssim = self.loss(predict, labels)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, loss_L, loss_ssim
