import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc

from visualize import *

import matplotlib.pyplot as plt

logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

class WassersteinGAN(object):
    def __init__(self, g_net, d_net, data, model):
        self.model = model
        self.dataset = data
        self.data = self.dataset.name
        self.g_net = g_net
        self.d_net = d_net

        self.x_sampler = self.dataset.train_sampler[0]
        self.y_sampler = self.dataset.train_sampler[1]
        self.name_sampler = self.dataset.train_sampler[2]
        self.z_sampler = self.dataset.noise_sampler

        self.batch_size = self.dataset.config.batch_size
        self.z_dim = self.dataset.config.z_dim
        self.image_size = self.dataset.config.image_size
        self.logdir = self.dataset.config.logdir

        self.x = tf.placeholder(tf.float32, self.x_sampler.get_shape(), name='x')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.r = tf.placeholder(tf.float32, [self.batch_size, 117], name='r')
        self.o = tf.placeholder(tf.float32, [self.batch_size, 80], name='o')

        self.x_ = self.g_net(self.z, self.r, self.o)
        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.d_[:, :80], self.o) \
                    + tf.nn.softmax_cross_entropy_with_logits(self.d_[:, 80:], self.r))
        self.d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.d[:, :80], self.o) \
                    + tf.nn.softmax_cross_entropy_with_logits(self.d[:, 80:], self.r)) \
                    - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.d_[:, :80], self.o) \
                    + tf.nn.softmax_cross_entropy_with_logits(self.d_[:, 80:], self.r))

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
            .minimize(self.d_loss_reg, var_list=self.d_net.vars)
        self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
            .minimize(self.g_loss_reg, var_list=self.g_net.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]

        # init some summary variables
        self.g_loss_monitor = tf.Variable(0.0, trainable = False)
        self.d_loss_monitor = tf.Variable(0.0, trainable = False)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss_monitor)
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss_monitor)
        self.merged = tf.summary.merge([self.g_loss_summary, self.d_loss_summary])
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        tf.train.start_queue_runners(sess=self.sess)

    def init_summary(self):
        self.path = 'logs/{}/{}/'.format(self.data, self.logdir)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.train_writer = tf.summary.FileWriter(self.path, self.sess.graph)

    def write_summary(self, d_loss, g_loss, t):
        self.sess.run(self.g_loss_monitor.assign(g_loss))
        self.sess.run(self.d_loss_monitor.assign(d_loss))
        [summary]  = self.sess.run([self.merged])
        self.train_writer.add_summary(summary, t)

    def terminate(self):
        self.train_writer.close()

    def process_label(self, y, sample=True):
        positives, negatives = [], []
        if sample:
            for i in range(y.shape[0]):
                positives.append(self.dataset.train.list_relation[np.where(y[i] == 1)[0][0]])
                negatives.append(self.dataset.train.list_relation[np.where(y[i] == -1)[0][0]])
        else:
            for i in range(y.shape[0]):
                positives.append(self.dataset.train.list_relation[np.where(y[i] == 1)[0]])
                negatives.append(self.dataset.train.list_relation[np.where(y[i] == -1)[0]])
        return positives, negatives 


    def dense_to_one_hot(self, labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def train(self, num_batches=1000000):
        self.init_summary()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            if t % 500 == 0 or t < 25:
                 d_iters = 100

            for _ in range(0, d_iters):
                bx, by, names = self.sess.run([self.x_sampler, self.y_sampler, self.name_sampler])
                by, bfy = self.process_label(by, sample=True)
                bo = self.dense_to_one_hot(np.array([int(ins[2]) for ins in by]), 80)
                br = self.dense_to_one_hot(np.array([int(ins[3]) for ins in by]), 117)
                bz = self.z_sampler(self.batch_size, self.z_dim)
                self.sess.run(self.d_clip)
                self.sess.run(self.d_rmsprop, feed_dict={self.x: bx, self.z: bz, self.o: bo, self.r: br})

            bx, by, names = self.sess.run([self.x_sampler, self.y_sampler, self.name_sampler])
            by, bfy = self.process_label(by, sample=True)
            # print([ins for ins in by])
            bo = self.dense_to_one_hot(np.array([int(ins[2]) for ins in by]), 80)
            br = self.dense_to_one_hot(np.array([int(ins[3]) for ins in by]), 117)
            bz = self.z_sampler(self.batch_size, self.z_dim)
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.o: bo, self.r: br})

            if t % 100 == 0 or t < 100:
                #bx = self.x_sampler(batch_size)
                bx, by = self.sess.run([self.x_sampler, self.y_sampler])
                by, bfy = self.process_label(by, sample=True)
                bo = self.dense_to_one_hot(np.array([int(ins[2]) for ins in by]), 80)
                br = self.dense_to_one_hot(np.array([int(ins[3]) for ins in by]), 117)
                bz = self.z_sampler(self.batch_size, self.z_dim)
                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz, self.o: bo, self.r: br})
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz, self.o: bo, self.r: br})
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t + 1, time.time() - start_time, d_loss, g_loss))
                self.write_summary(d_loss, g_loss, t)

            if t % 100 == 0:
                bz = self.z_sampler(self.batch_size, self.z_dim)
                by = self.sess.run(self.y_sampler)
                by, bfy = self.process_label(by, sample=True)
                bo = self.dense_to_one_hot(np.array([int(ins[2]) for ins in by]), 80)
                br = self.dense_to_one_hot(np.array([int(ins[3]) for ins in by]), 117)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz, self.o: bo, self.r: br})
                rescaled = np.divide(bx + 1.0, 2.0)
                np.reshape(np.clip(rescaled, 0.0, 1.0), bx.shape)
                y = concat_multiple_images(bx)
                #bx = xs.data2img(bx)
                scipy.misc.imsave('{}/{}.png'.format(self.path, t), y)

        self.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='hico')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--datadir', type=str, default='/mfs/zhijie/data/tfrecord')
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--z_dim', type=int, default=100)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    model = importlib.import_module(args.data + '.' + args.model)
    config = importlib.import_module(args.data).config(
                batch_size = args.batch_size, image_size = args.image_size,
                logdir = args.logdir, z_dim = args.z_dim,
                datadir = args.datadir)

    data = importlib.import_module(args.data).dataset(config)

    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, data, args.model)
    wgan.train()
