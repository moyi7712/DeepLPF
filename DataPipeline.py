import tensorflow as tf
import numpy as np
import json, os
import cv2


class Data(object):
    def __init__(self, config, batch_size=1):
        self.config = config
        self.batch_size = batch_size
        self.resize_method = int(getattr(cv2, self.config.resize_method))

        self.pad_method = self.config.resize_pad_method
        with open(self.config.filelist, 'r') as f:
            self.filelist = json.load(f)

    def pipeline(self):

        dataset = tf.data.Dataset.from_tensor_slices(self.filelist['train'])
        dataset = dataset.map(lambda path: tf.numpy_function(self.np_read, [path], [tf.float32, tf.float32]), num_parallel_calls=6)
        dataset = dataset.map(self.enhance, num_parallel_calls=6)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.shuffle(self.config.shuffle).prefetch(buffer_size=self.config.prefetch)

        return dataset

    def np_read(self, path):
        path = str(path, encoding='utf-8')
        origin_path = os.path.join(self.config.origin_path, path + '.' + self.config.image_extension)
        label_path = os.path.join(self.config.label_path, path + '.' + self.config.image_extension)
        origin = cv2.cvtColor(cv2.imread(origin_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        w, h, _ = origin.shape
        w, h = tuple(self.config.longest_pixs * np.array([w, h]) / max(w, h))
        w, h = int(w), int(h)

        origin = cv2.resize(origin, (w, h), interpolation=self.resize_method)
        label = cv2.resize(label, (w, h), interpolation=self.resize_method)
        if self.config.is_resize:
            pad_or_resize = np.random.randint(0, 2)
            if pad_or_resize:
                # resize
                origin = cv2.resize(origin, (self.config.longest_pixs, self.config.longest_pixs),
                                    interpolation=self.resize_method)
                label = cv2.resize(label, (self.config.longest_pixs, self.config.longest_pixs),
                                   interpolation=self.resize_method)
            else:
                # padding
                padding_w = np.random.randint(0, self.config.longest_pixs - w + 1)
                padding_h = np.random.randint(0, self.config.longest_pixs - h + 1)
                pad_method = np.random.choice(self.pad_method)
                padding = [
                    [padding_h, self.config.longest_pixs - h - padding_h],
                    [padding_w, self.config.longest_pixs - w - padding_w],
                    [0, 0]]
                origin = np.pad(origin, padding, mode=pad_method)
                label = np.pad(label, padding, mode=pad_method)

        for i in range(3):
            assert np.shape(origin)[i] == np.shape(label)[i]
        origin = np.float32(origin)/255
        label = np.float32(label)/255
        return origin, label

    @tf.function
    def flip(self, origin, label):
        if self.config.is_flip:
            if np.random.randint(0, 1) == 0:
                origin = tf.image.flip_up_down(origin)
                label = tf.image.flip_up_down(label)
            if np.random.randint(0, 1) == 0:
                origin = tf.image.flip_left_right(origin)
                label = tf.image.flip_left_right(label)
        return origin, label

    @tf.function
    def rot(self, origin, label):
        if self.config.is_rot:
            rot_angle = np.random.randint(0, 5)
            origin = tf.image.rot90(origin, rot_angle)
            label = tf.image.rot90(label, rot_angle)
        return origin, label

    def enhance(self, origin, label):

        origin, label = self.flip(origin, label)
        origin, label = self.rot(origin, label)

        return origin, label
