import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

from layers import *


class Discriminator(object):
    def __init__(self):
        self.name = 'hico/rgan/d_net'

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
        self.name = 'hico/rcan/n_net'

    def __call__(self, noise, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variable()
            bs = tf.shape(noise)[0]
            fc = tcl.fully_connected(noise, 4 * 4 * 1024, activation_fn = tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)
            conv2 = tcl.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm)
            conv3 = tcl.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm)
            return conv3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class ObjectGenerator(object):
    def __init__(self):
        self.name = 'hico/rgan/o_net'

    def __call__(self, obj, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(obj)[0]
            num_obj = tf.shape(obj)[1]
            fc = tcl.fully_connected(obj, 4 * 4 * 1024, activation_fn = tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)
            conv2 = tcl.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm)
            conv3 = tcl.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm)
            return conv3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class RelationGenerator(object):
    def __init__(self):
    	self.name = 'hico/rgan/r_net'
    
    def __call__(self, obj_feature, noise_feature, r, reuse=True):
    	with tf.variable_scope(self.name) as vs:
    	    if reuse:
    	        vs.reuse_variables()
            bs = tf.shape(r)[0]
            fc = tcl.fully_connected(r, 16 * 16 * 512, activation_fn = tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 16, 16, 512]))
            conv1 = relu_batch_norm(conv1)
            conv1 = tf.concat([noise_feature, conv1, obj_feature], 3)
            print(conv1)
            conv2 = tcl.conv2d(conv1, 512, [5, 5], [1, 1], 
                weights_initializer = tf.random_normal_initializer(stddev=0.02),
                activation_fn = relu_batch_norm)
            conv3 = tcl.conv2d(conv2, 256, [5, 5], [1, 1],
                weights_initializer = tf.random_normal_initializer(stddev=0.02),
                activation_fn = relu_batch_norm)
            return conv3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator(object):
    def __init__(self):
        self.name = 'hico/rgan/g_net'
        self.o_net = ObjectGenerator()
        self.n_net = NoiseGenerator()
        self.r_net = RelationGenerator() 

    def __call__(self, z, r, o):
        with tf.variable_scope(self.name) as vs:
            noise_feature = self.n_net(z, reuse = False)
            obj_feature = self.o_net(o, reuse = False)

            # fused is in dimension [batch_size, M/4, M/4, 256]
            fused_maps = self.r_net(noise_feature, obj_feature, r, reuse = False)

            conv1 = tcl.conv2d_transpose(
                fused_maps, 128, [4, 4], [2, 2],
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
