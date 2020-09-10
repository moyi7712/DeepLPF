import tensorflow as tf
import tensorflow.keras as keras
import math
import numpy as np
from Utils import sign01, tanh01
from Layers import Conv as FilterConv
from BackBone import Unet


class Filters(keras.models.Model):
    def __init__(self, channels, FC_num, name):
        super(Filters, self).__init__()
        self.layer_1 = FilterConv(channels=channels, name=name + '_1')
        self.layer_2 = FilterConv(channels=channels, name=name + '_2')
        self.layer_3 = FilterConv(channels=channels, name=name + '_3')
        self.layer_4 = FilterConv(channels=channels, name=name + '_4')
        self.factor_num = FC_num
        self.fc = keras.layers.Dense(self.factor_num)
        self.dropout = keras.layers.Dropout(0.5)
        self.upsample_size = 300
        self.upsample_mode = 'BILINEAR'

    def get_axis(self, w, h):
        axis_x = np.float32(np.repeat(np.expand_dims(np.arange(w), axis=1), h, axis=1)) / w
        axis_y = np.float32(np.repeat(np.expand_dims(np.arange(h), axis=0), w, axis=0)) / h
        axis_x = np.expand_dims(axis_x, axis=2)
        axis_y = np.expand_dims(axis_y, axis=2)
        axis = np.concatenate((axis_x, axis_y), axis=2)
        axis_tf = tf.expand_dims(tf.convert_to_tensor(axis, dtype=tf.float32), axis=0)
        return axis_tf


class CubicFilter(Filters):
    def __init__(self):
        super(CubicFilter, self).__init__(channels=64, FC_num=60, name='Cubic')

    def get_cubic20_mask(self, image, axis_tf):
        _, w, h, _ = tf.shape(image)

        R, G, B = tf.split(image, 3, axis=3)
        axis_tf = tf.concat(values=[axis_tf, tf.ones_like(R)], axis=3)
        channel_R = tf.concat(values=[axis_tf, R], axis=3)
        channel_G = tf.concat(values=[axis_tf, G], axis=3)
        channel_B = tf.concat(values=[axis_tf, B], axis=3)
        axis_mul = tf.expand_dims(axis_tf[:, :, :, 0] * axis_tf[:, :, :, 1], axis=3)

        cubic_R_1 = tf.reshape(tf.einsum('ijkl,ijkn->ijkln', channel_R, channel_R ** 2), (1, w, h, 16))
        cubic_G_1 = tf.reshape(tf.einsum('ijkl,ijkn->ijkln', channel_G, channel_G ** 2), (1, w, h, 16))
        cubic_B_1 = tf.reshape(tf.einsum('ijkl,ijkn->ijkln', channel_B, channel_B ** 2), (1, w, h, 16))
        cubic_R_2 = axis_mul * R / (channel_R + 1e-8)
        cubic_G_2 = axis_mul * G / (channel_G + 1e-8)
        cubic_B_2 = axis_mul * B / (channel_B + 1e-8)
        cubic_R = tf.concat(values=[cubic_R_1, cubic_R_2], axis=3)
        cubic_G = tf.concat(values=[cubic_G_1, cubic_G_2], axis=3)
        cubic_B = tf.concat(values=[cubic_B_1, cubic_B_2], axis=3)
        return cubic_R, cubic_G, cubic_B

    def call(self, inputs, training=None, mask=None):
        feat, img = inputs
        b, w, h, _ = tf.shape(img)

        feat_cubic = tf.concat(values=[feat, img], axis=3)

        feat_cubic = tf.image.resize(feat_cubic, (self.upsample_size, self.upsample_size),
                                     method=getattr(tf.image.ResizeMethod, self.upsample_mode))

        # network
        x = self.layer_1(feat_cubic)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.dropout(x)
        factor = self.fc(x)
        factor = tf.squeeze(factor)
        factor = tf.reshape(factor, (b, 1, self.factor_num, 1))

        # operator
        kernel_R, kernel_G, kernel_B = tf.split(factor, 3, axis=2)
        axis_tf = self.get_axis(w, h)
        cubic_mask_list = []

        for i in range(b):
            cubic_R, cubic_G, cubic_B = self.get_cubic20_mask(img[i:i + 1, :, :, :], axis_tf)

            mask_R = tf.nn.conv2d(cubic_R, kernel_R[i:i + 1, :, :, :], strides=1, padding='VALID')
            mask_G = tf.nn.conv2d(cubic_G, kernel_G[i:i + 1, :, :, :], strides=1, padding='VALID')
            mask_B = tf.nn.conv2d(cubic_B, kernel_B[i:i + 1, :, :, :], strides=1, padding='VALID')

            cubic_mask_list.append(tf.concat(values=[mask_R, mask_G, mask_B], axis=3))
        cubic_mask = tf.concat(values=cubic_mask_list, axis=0)

        return cubic_mask


class GraduatedFilter(Filters):
    def __init__(self):
        super(GraduatedFilter, self).__init__(channels=64, FC_num=24, name='Graduated')
        self.max_scale = 2

    def get_channel_mask(self, s, ginv, d1, d2, top_line):
        b, _, _, _ = tf.shape(ginv)

        f = tf.where(ginv == 0, (s - 1) / (2 * d1) + (s - 1) / (2 * d2),
                     (s - 1) / (2 * d1) + (1 - s) / (2 * d2))
        grad = tf.where(ginv == 0, s, 1)
        mask_list = []

        for i in range(b):
            clip_min, clip_max = tf.cond(s[i, 0, 0, 0] >= 1, lambda: (1, self.max_scale), lambda: (0, 1))
            mask = tf.clip_by_value(f[i:i+1, :, :, :] + grad[i:i + 1, :, :, :] * top_line[i:i + 1, :, :, :], clip_min,
                                    clip_max)
            mask_list.append(tf.clip_by_value(mask, 0, self.max_scale))

        return tf.concat(values=mask_list, axis=0)

    def get_Graguated_mask(self, factor, axis_tf):
        sr, sg, sb, m, c, o1, o2, g = tf.split(factor, 8, axis=1)
        axis_x, axis_y = tf.split(axis_tf, 2, axis=3)
        ginv = sign01(g)
        slope_angle = tf.atan(m)
        c = tanh01(c) + 1e-10
        o1 = tf.clip_by_value(sign01(o1), c, 1)
        o2 = tf.clip_by_value(sign01(o2), 0, c)
        sr = tanh01(sr) * self.max_scale
        sg = tanh01(sg) * self.max_scale
        sb = tanh01(sb) * self.max_scale
        d1 = tanh01(o1) * tf.cos(slope_angle)
        d2 = tanh01(o2) * tf.cos(slope_angle)
        top_line = tanh01(axis_y - (m * axis_x + c + d1))

        mask_r = self.get_channel_mask(sr, ginv, d1, d2, top_line)
        mask_g = self.get_channel_mask(sg, ginv, d1, d2, top_line)
        mask_b = self.get_channel_mask(sb, ginv, d1, d2, top_line)
        mask = tf.concat(values=[mask_r, mask_g, mask_b], axis=3)
        return mask

    def call(self, inputs, training=None, mask=None):
        feat, img = inputs
        b, w, h, _ = tf.shape(img)
        axis_tf = self.get_axis(w, h)
        axis_tf = tf.repeat(axis_tf, b, axis=0)
        feat_graduated = tf.concat(values=[feat, img], axis=3)
        feat_graduated = tf.image.resize(feat_graduated, (self.upsample_size, self.upsample_size),
                                         method=getattr(tf.image.ResizeMethod, self.upsample_mode))
        x = self.layer_1(feat_graduated)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.dropout(x)
        factor = self.fc(x)
        factor = tf.reshape(factor, (b, self.factor_num, 1, 1))

        factor_R, factor_G, factor_B = tf.split(factor, 3, axis=1)

        mask_r = self.get_Graguated_mask(factor_R, axis_tf)
        mask_g = self.get_Graguated_mask(factor_G, axis_tf)
        mask_b = self.get_Graguated_mask(factor_B, axis_tf)

        mask = tf.clip_by_value(mask_r * mask_g * mask_b, 0, self.max_scale)
        return mask


class EllipticalFilter(Filters):
    def __init__(self):
        super(EllipticalFilter, self).__init__(channels=64, FC_num=24, name='Elliptical')
        self.max_scale = 2
        self.eps = 1e-10

    def get_channel_mask(self, cond, sc, radius, m):
        mask = tf.where(cond < 1, m * (1 - sc) / radius + sc, 1)
        mask = tf.clip_by_value(mask, 0, self.max_scale)
        return mask

    def get_Elliptical_mask(self, factor, axis_tf):
        sr, sg, sb, h, k, theta, a, b = tf.split(factor, 8, axis=1)

        axis_x, axis_y = tf.split(axis_tf, 2, axis=3)

        sr = sr * self.max_scale + self.eps
        sg = sg * self.max_scale + self.eps
        sb = sb * self.max_scale + self.eps
        h, k, a, b = h + self.eps, k + self.eps, a + self.eps, b + self.eps
        theta = theta * math.pi + self.eps

        angle = tf.acos(
            tf.clip_by_value(
                (axis_y - k) / (tf.sqrt((axis_x - h) ** 2 + (axis_y - k) ** 2 + self.eps) + self.eps),
                -1 + 1e-7, 1 - 1e-7)) - theta
        radius = ((a * b) / (tf.sqrt(
            (a ** 2) * (tf.sin(angle) ** 2) + (b ** 2) * (tf.cos(angle) ** 2) + self.eps) + self.eps)) + self.eps

        cond = (((((axis_x - h) * tf.cos(theta) + (axis_y - k) * tf.sin(theta)) ** 2) / (a ** 2)) +
                ((((axis_x - h) * tf.sin(theta) - (axis_y - k) * tf.cos(theta)) ** 2) / (b ** 2)) + self.eps)

        scale_m = tf.sqrt((axis_x - h) ** 2 + (axis_y - k) ** 2 + self.eps)
        mask_r = self.get_channel_mask(cond, sr, radius, scale_m)
        mask_g = self.get_channel_mask(cond, sg, radius, scale_m)
        mask_b = self.get_channel_mask(cond, sb, radius, scale_m)
        mask = tf.clip_by_value(tf.concat(values=[mask_r, mask_g, mask_b], axis=3), 0, self.max_scale)
        return mask

    def call(self, inputs, training=None, mask=None):
        feat, img = inputs
        b, w, h, _ = tf.shape(img)
        axis_tf = self.get_axis(w, h)
        axis_tf = tf.repeat(axis_tf, b, axis=0)
        feat_elliptical = tf.concat(values=[feat, img], axis=3)
        feat_elliptical = tf.image.resize(feat_elliptical, (self.upsample_size, self.upsample_size),
                                          method=getattr(tf.image.ResizeMethod, self.upsample_mode))
        x = self.layer_1(feat_elliptical)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.dropout(x)
        factor = self.fc(x)
        factor = tf.reshape(factor, (b, self.factor_num, 1, 1))
        factor = tanh01(factor)
        factor_R, factor_G, factor_B = tf.split(factor, 3, axis=1)

        mask_R = self.get_Elliptical_mask(factor_R, axis_tf)
        mask_G = self.get_Elliptical_mask(factor_G, axis_tf)
        mask_B = self.get_Elliptical_mask(factor_B, axis_tf)
        mask = tf.clip_by_value(mask_R * mask_G * mask_B, 0, self.max_scale)
        return mask


class DeepLPF(keras.models.Model):
    def __init__(self):
        super(DeepLPF, self).__init__()
        self.backbone = Unet()
        self.cubic = CubicFilter()
        self.graduated = GraduatedFilter()
        self.elliptical = EllipticalFilter()

    def call(self, inputs, training=None, mask=None):
        img = inputs
        feat = self.backbone(img)
        cubic_mask = self.cubic(inputs=(feat, img))

        graduated_mask = self.graduated(inputs=(feat, img))
        elliptical_mask = self.elliptical(inputs=(feat, img))

        mask_fuse = tf.clip_by_value(graduated_mask + elliptical_mask, 0, 2)
        img_fuse = tf.clip_by_value(cubic_mask * mask_fuse, 0, 1)
        output = tf.clip_by_value(img_fuse + img, 0, 1)
        return output
