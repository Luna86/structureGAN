import importlib
import tensorflow as tf
import tensorflow.contrib as tc
#import matplotlib.pyplot as plt
import os
from visualize import *
import numpy as np
import scipy.misc


logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)


image_size = 256

config = importlib.import_module('hico').config(
                batch_size = 16, image_size = 64,
				datadir = '/datasets/BigLearning/hzhang2/data/hico')

data = importlib.import_module('hico').dataset(config)

read_op, label_op, _ = data.train_sampler
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)


sess.run(tf.global_variables_initializer())
bx, labels = sess.run([read_op, label_op])

print('reach here..')
positives = []
negatives = []
for i in range(labels.shape[0]):
    positives.append(data.train.list_relation[np.where(labels[i] == 1)[0]])
    negatives.append(data.train.list_relation[np.where(labels[i] == -1)[0]])
#for i in range(labels.shape[0]):
#    positives.append(data.train.list_relation[np.where(labels[i] == 1)])
#    negatives.append(data.train.list_relation[np.where(labels[i] == -1)])
print(positives)
#print(negatives)
        
#convert_op = tf.image.convert_image_dtype(bx, dtype=tf.uint8)
#bx = sess.run(convert_op)
#y = concat_multiple_images(bx)
#path = 'logs/{}/'.format('test')
#if not os.path.exists(path):
#    os.makedirs(path)
#scipy.misc.imsave('logs/{}/{}.png'.format('test', 1), y)
#fig = plt.figure('dcgan')
#grid_show(fig, bx, [image_size, image_size, 3])
#path = 'logs/{}/'.format('test')
#if not os.path.exists(path):
#    os.makedirs(path)
#fig.savefig('logs/{}/{}.png'.format('test', 1))
