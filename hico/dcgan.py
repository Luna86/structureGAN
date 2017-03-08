import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

from layers import *

class Discriminator(object):
    def __init__(self):
        #self.x_dim = 64 * 64 * 3
        self.name = 'hico/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            #x = tf.reshape(x, [bs, 64, 64, 3])
            conv1 = tcl.conv2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu
            )
            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv3 = tcl.conv2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.conv2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.flatten(conv4)
            fc = tcl.fully_connected(conv4, 1, activation_fn=tf.identity)
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class NoiseGenerator(object):
    def __init__(self):
        self.name = 'hico/dcgan/ng_net'
        self.n_dim = 100
    def __call__(self, noise, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(noise)[0]
            fc = tcl.fully_connected(noise, 4 * 4 * 1024, activation_fn=tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)
            conv2 = tcl.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv3 = tcl.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            return conv3
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class ObjectGenerator(object):
    def __init__(self):
        self.name = 'hico/dcgan/og_net'
        self.o_dim = 80
    def __call__(self, obj, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(obj)[0]
            fc = tcl.fully_connected(obj, 4 * 4 * 1024, activation_fn=tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 1024]))
            conv1 = tf.nn.relu(conv1)
            conv2 = tcl.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu
            )
            conv3 = tcl.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu
            )
            return conv3
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class RelationshipComposition(object):
    def __init__(self):
        self.name = 'hico/dcgan/rc_net'
        self.r_dim = 117
    def __call__(self, feature1, feature2, rela, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(rela)[0]
            fc = tcl.fully_connected(rela, 512, activation_fn=tf.nn.relu)
            feature = tf.reshape(tf.tile(fc, 16 * 16), tf.stack([bs, 16, 16, 512]))
            # feature = relu_batch_norm(feature)
            conv1 = tcl.conv2d(
                tf.concat(3, [feature1, feature, feature2]), 512, [5, 5], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv2 = tcl.conv2d(
                conv1, 256, [5, 5], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            # rela can generate a feature map
            # W_conv = tf.get_variable('W_conv_all', [self.r_dim, 5, 5, 512, 384], tf.float32, tf.random_normal_initializer(stddev=0.02))
            # b_conv = tf.get_variable('b_conv_all', [self.r_dim, 256], tf.float32, tf.constant_initializer(0.0, dtype=tf.float32))
            # W_conv_1 = tf.get_variable('W_conv_all_1', [self.r_dim, 5, 5, 384, 256], tf.float32, tf.random_normal_initializer(stddev=0.02))
            # b_conv_1 = tf.get_variable('b_conv_all_1', [self.r_dim, 256], tf.float32, tf.constant_initializer(0.0, dtype=tf.float32))
            # conv1 = relu_batch_norm(\
            #         tf.nn.conv2d(tf.concat(3, [feature1, feature2]), tf.reduce_sum(W_conv * rela, 0), strides=[1, 1, 1, 1], padding='SAME')\
            #         + tf.reduce_sum(b_conv * rela, 0)\
            #         )
            # conv2 = relu_batch_norm(\
            #         tf.nn.conv2d(conv1, tf.reduce_sum(W_conv_1 * rela, 0), strides=[1, 1, 1, 1], padding='SAME')\
            #         + tf.reduce_sum(b_conv_1 * rela, 0)\
            #         )
            return conv2
            return tf.concat(3, [feature1, feature, feature2])
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self):
        #self.z_dim = 100
        #self.x_dim = 64 * 64 * 3
        self.name = 'hico/dcgan/g_net'
        self.og = ObjectGenerator()
        self.rc = RelationshipComposition()
        self.ng = NoiseGenerator()

    def __call__(self, z, r, o):
        with tf.variable_scope(self.name) as vs:
            feature1 = self.ng(z, reuse=False)
            feature2 = self.og(tf.one_hot(o, 80), reuse=False)
            feature = self.rc(feature1, feature2, tf.one_hot(r, 117), reuse=False)
            conv1 = tcl.conv2d_transpose(
                feature, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv2 = tcl.conv2d_transpose(
                conv1, 3, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.tanh)
            return conv2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
