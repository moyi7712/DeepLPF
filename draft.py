import tensorflow as tf
from Models import DeepLPF
import Utils
img_0 = tf.random.uniform(shape=(1, 512, 512, 3), dtype=tf.float32, maxval=1)
img_1 = tf.random.uniform(shape=(1, 512, 512, 3), dtype=tf.float32, maxval=1)

a1,_,_ = tf.split(img_0, 3, axis=3)
a2,_,_ = tf.split(img_1, 3, axis=3)

ms_ssim = tf.image.ssim_multiscale(a1,a2, max_val=1.0, filter_size=5, filter_sigma=0.5)
print(ms_ssim)
