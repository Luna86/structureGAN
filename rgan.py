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

class RGAN(object):
    def __init__(self, g_net, d_net, data, model):
        self.model = model
        self.dataset = data
        self.g_net = g_net
        self.d_net = d_net

        self.x_sampler = self.dataset.train_sampler[0]
        self.index_sampler = self.dataset.train_sampler[1]
        self.h_sampler = self.dataset.train_sampler[3]
        self.w_sampler = self.dataset.train_sampler[4]
        self.z_sampler = self.dataset.noise_sampler

        self.batch_size = self.dataset.config.batch_size
        self.z_dim = self.dataset.config.z_dim
        self.image_size = self.dataset.config.image_size
        self.logdir = self.dataset.config.logdir

        self.num_action = self.dataset.train.num_action
        self.num_obj = self.dataset.train.num_obj
        self.list_action = self.dataset.train.list_action
        self.list_obj = self.dataset.train.list_obj
        self.list_relation = self.dataset.train.list_relation

        self.x = tf.placeholder(tf.float32, self.x_sampler.get_shape(), name='x')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        # caz Hico's relations are all human -> action -> object
        self.r = tf.placeholder(tf.float32, [self.batch_size, self.num_action], name = 'r') 
        self.o = tf.placeholder(tf.float32, [self.batch_size, self.num_obj], name = 'o') 

        self.x_ = self.g_net(self.z, self.r, self.o)
        self.d = self.d_net(self.x, reuse=False) 
        self.d_ = self.d_net(self.x_)

        #TODO
        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

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
        self.path = 'logs/{}/{}/'.format(self.dataset.name, self.logdir)
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
            for i in range(y.shape[1]):
                positives.append(self.list_relation[np.where(y[:, i] == 1)[0][0]])
                negatives.append(self.list_relation[np.where(y[:, i] == -1)[0][0]])
        else:
            for i in range(y.shape[0]):
                positives.append(self.list_relation[np.where(y[:, i] == 1)[0]])
                negatives.append(self.list_relation[np.where(y[:, i] == -1)[0]])
        return positives, negatives
    
    def dense_to_one_hot(self, labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros([num_labels, num_classes])
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def sample_label_and_bboxes(self, indices):
        assert(indices.shape[0] == self.batch_size)
        y = self.dataset.train.label[:, indices]
        raw_label, _ = self.process_label(y, sample = True)
        o_dense_label = np.array([int(ins[2]) for ins in raw_label])
        r_dense_label = np.array([int(ins[3]) for ins in raw_label])
        bo = self.dense_to_one_hot(o_dense_label, self.num_obj)
        br = self.dense_to_one_hot(r_dense_label, self.num_action)
       
        bboxes = []
        for i in range(self.batch_size):
            bboxes.append(self.dataset.train.bboxes[indices[i]])
        # sample the first connection of the first relation
        sample = True
        if sample:
            h_box, o_box = np.zeros([self.batch_size, 4]), np.zeros([self.batch_size, 4])
            for i in range(self.batch_size):
                if len(bboxes[i][0]) > 0: # if the number connections is 0, then set their bounding boxes to zero
                    h_box[i,:], o_box[i, :] = bboxes[i][0][0][0, :], bboxes[i][0][0][1, :]
        return bo, br, h_box, o_box
      
    #compute the relative position of the bboxes after resizing.
    def map_bboxes(self, h_box, o_box, h, w):
        # infer the relative location of h_box and o_box
        wwhh = np.reshape(np.concatenate([h, h, w, w], axis = 0), [h.shape[0], 4])
        h_box = h_box / wwhh * self.image_size
        o_box = o_box / wwhh * self.image_size
        return np.floor(h_box), np.floor(o_box)

    def train(self, num_batches=1000000):
        self.init_summary()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            if t % 500 == 0 or t < 25:
                 d_iters = 100

            for _ in range(0, d_iters):
                bx, bx_index, height, width = self.sess.run([self.x_sampler, 
                        self.index_sampler, self.h_sampler, self.w_sampler])
                bo, br, h_box, o_box = self.sample_label_and_bboxes(bx_index) # sample the first relation, and the first connection of this relation

                h_box, o_box = self.map_bboxes(h_box, o_box, height, width)

                bz = self.z_sampler(self.batch_size, self.z_dim)
                self.sess.run(self.d_clip)
                self.sess.run(self.d_rmsprop, 
                    feed_dict={self.x: bx, self.z: bz, self.o: bo, self.r: br})

            bz = self.z_sampler(self.batch_size, self.z_dim)
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.o: bo, self.r: br})

            if t % 100 == 0 or t < 100:
                #bx = self.x_sampler(batch_size)
                bx, bx_index = self.sess.run([self.x_sampler, self.index_sampler])
                bo, br = self.sample_label(bx_index)
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
                x_gt, bx_index = self.sess.run([self.x_sampler, self.index_sampler])
                bo, br = self.sample_label(bx_index)
                x_gen = self.sess.run(self.x_, feed_dict={self.z: bz, self.o: bo, self.r: br})
                rescaled = np.divide(x_gen + 1.0, 2.0)
                np.reshape(np.clip(rescaled, 0.0, 1.0), x_gen.shape)
                x_gen = concat_multiple_images(x_gen)
                x_gt = concat_multiple_images(x_gt)
                #bx = xs.data2img(bx)
                scipy.misc.imsave('{}/{}.png'.format(self.path, t), x_gen)
                scipy.misc.imsave('{}/gt-{}.png'.format(self.path, t), x_gt)

        self.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='hico')
    parser.add_argument('--model', type=str, default='rgan')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--datadir', type=str, default='/datasets/BigLearning/hzhang2/data/hico_det')
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
    rgan = RGAN(g_net, d_net, data, args.model)
    rgan.train()
