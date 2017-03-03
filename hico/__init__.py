import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from hico.hico_data import HicoData 
from hico import image_processing

num_preprocess_threads = 4
batchsize = 32


class dataset(object):
    def __init__(self):
        self.train = HicoData(subset = 'train') 
        self.test = HicoData(subset = 'test')
        self.train_sampler = image_processing.distorted_inputs(
                                self.train, batch_size = batchsize, 
                                num_preprocess_threads = num_preprocess_threads)

    class NoiseSampler(object):
        def __call__(self, batch_size, z_dim):
            return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])


#class DataSampler(object):
#    def __init__(self):
#        self.shape = [28, 28, 1]
#
#    def __call__(self, batch_size):
#        return mnist.train.next_batch(batch_size)[0]
#
#    def data2img(self, data):
#        return np.reshape(data, [data.shape[0]] + self.shape)
#
#
