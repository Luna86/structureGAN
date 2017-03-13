import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from hico.hico_data import HicoData 
from hico import image_processing

class config(object):
    def __init__(self, batch_size=32, image_size=64, num_preprocess_threads=4, 
									z_dim=100, logdir='', datadir = ''):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_preprocess_threads = num_preprocess_threads
        self.z_dim = z_dim
        self.logdir = logdir
        self.datadir = datadir

class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])

class dataset(object):
    def __init__(self, config):
        self.config = config
        self.train = HicoData(subset = 'train', datadir = self.config.datadir)
        #self.test = HicoData(subset = 'test', data_dir = self.config.data_dir)
        self.name = 'hico'

        self.train_sampler = image_processing.distorted_inputs(
                                self.train, batch_size = self.config.batch_size,
                                image_size = self.config.image_size, 
                                num_preprocess_threads = self.config.num_preprocess_threads)
        self.noise_sampler = NoiseSampler()


