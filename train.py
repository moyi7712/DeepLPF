import tensorflow as tf
import tensorflow.keras as keras
from Utils import Config
from Network import Net
from DataPipeline import Data
import os

config = Config('./config.yaml')
data_config = config.Dataset()
net_config = config.HyperPara()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
net = Net(net_config)
data = Data(data_config)
train_data = data.train()
valid_data = data.valid()

for epoch in range(net_config.epoch):
    index = 0
    loss_sum = 0
    for origin, label in train_data:
        loss, loss_L, loss_ssim = net.train(origin, label)
        loss_sum += loss.numpy()
        # loss_L = loss_L.numpy()
        # loss_ssim = loss_ssim.numpy()
        # loss_L_sum += loss_L
        # loss_ssim_sum += loss_ssim
        # loss_sum += loss
        if index % 100 == 0:
            net.ckpt_manager.save()
            print("INFO: epoch{}, step:{}, loss:{}".format(epoch, index, loss_sum/100))
            loss_sum = 0
        index += 1
    # loss_ssim_sum /= index
    # loss_L_sum /= index
    # loss_sum /= index
    # with net.summary_writer.as_default():
    #     tf.summary.scalar('loss_ssim', loss_ssim_sum, step=epoch)
    #     tf.summary.scalar('loss_l1', loss_L_sum, step=epoch)
    #     tf.summary.scalar('loss_total', loss_sum, step=epoch)
    # psnr = 0
    # index = 0
    #
    # for origin, label in valid_data:
    #     predict = net.model(origin)
    #     loss, _, _ = net.loss(predict, label)
    #     psnr += tf.reduce_mean(tf.image.psnr(predict, label, max_val=1.0)).numpy()
    #     index += 1
    # psnr /= index
    # with net.summary_writer.as_default():
    #     tf.summary.scalar('test_psnr', psnr, step=epoch)
    # print('INFO: loss:{}, psnr{}'.format(loss_sum, psnr))
    #
    #
