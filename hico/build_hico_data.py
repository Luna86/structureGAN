from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import scipy.io
import h5py
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train_directory', '/datasets/BigLearning/hzhang2/data/hico_det/hico_20160224_det/images/train2015/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/datasets/BigLearning/hzhang2/data/hico_det/hico_20160224_det/images/test2015/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/datasets/BigLearning/hzhang2/data/hico_det/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 16,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 4,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, index, 
                        height, width):

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/index': _int64_feature(index),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _process_image(filename, coder):
  with tf.gfile.FastGFile(filename, 'r') as f:
      image_data = f.read()
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                                arr, num_shards):
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      index = arr[i]
      image_buffer, height, width = _process_image(filename, coder)
      example = _convert_to_example(filename, image_buffer, index, 
                                    height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, arr, num_shards):

  assert len(filenames) == arr.shape[0]
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, arr, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()

def _process_dataset(name, directory, num_shards, list_image, arr):
  filenames = []
  for i in list_image:
    path = os.path.join(directory, i)
    filenames.append(path)
  _process_image_files(name, filenames, arr, num_shards)

def _read_annotations():
  datadir = '/datasets/BigLearning/hzhang2/data/hico_det/hico_20160224_det/'
  anno_file = os.path.join(datadir, 'anno.mat')
  mat = scipy.io.loadmat(anno_file) 
  list_train_raw, list_test_raw = mat['list_train'], mat['list_test']
  list_train, list_test = [], []
  for im_name in list_train_raw:
    list_train.append(im_name[0][0])
  for im_name in list_test_raw:
    list_test.append(im_name[0][0])

  # random shuffle the data
  train_arr = np.arange(len(list_train))
  np.random.shuffle(train_arr)
  list_train_shuf= []
  for i in range(train_arr.shape[0]):
    list_train_shuf.append(list_train[train_arr[i]])
  
  test_arr = np.arange(len(list_test))
  np.random.shuffle(test_arr)
  list_test_shuf = []
  for i in range(test_arr.shape[0]):
    list_test_shuf.append(list_test[test_arr[i]])

  return list_train_shuf, list_test_shuf, train_arr, test_arr

def _from_unicode_to_ascii(list_train, list_test):
  #list_obj = [n.encode("ascii", "ignore") for n in list_obj]
  #list_action = [n.encode("ascii", "ignore") for n in list_action]
  list_train = [n.encode("ascii", "ignore") for n in list_train]
  list_test = [n.encode("ascii", "ignore") for n in list_test]
  #for i in list_relation:
  #  i[0] = i[0].encode("ascii", "ignore")
  #  i[1] = i[1].encode("ascii", "ignore")
  #return list_obj, list_action, list_train, list_test, list_relation
  return list_train, list_test

def _save_meta(list_obj, list_action, list_train, list_test, list_relation):
  # first, save the metadata into hdf5
  datadir = '/datasets/BigLearning/hzhang2/data/hico_det/hico_20160224_det/'
  meta = h5py.File(os.path.join(datadir, 'meta.h5'), 'w')
  meta.create_dataset('list_obj', data = list_obj)
  meta.create_dataset('list_action', data = list_action)
  meta.create_dataset('list_relation', data = list_relation)
  meta.create_dataset('list_train', data = list_train)
  meta.create_dataset('list_test', data = list_test)
  meta.close()

def _save_label(train_label, test_label, bboxes_train, bboxes_test):
  datadir = '/datasets/BigLearning/hzhang2/data/hico_det/hico_20160224_det/'
  meta = h5py.File(os.path.join(datadir, 'label.h5'), 'w')
  meta.create_dataset('train_label', data = train_label)
  meta.create_dataset('test_label', data = test_label)
  meta.create_dataset('train_bboxes', data = bboxes_train)
  meta.create_dataset('test_bboxes', data = bboxes_test)
  meta.close()
  return

def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  list_train, list_test, train_arr, test_arr = _read_annotations()

  # Run it!
  list_train, list_test = _from_unicode_to_ascii(list_train, list_test)
  #_save_meta(list_obj, list_action, list_train, list_test, list_relation)
  #_save_label(train_label, test_label, bboxes_train, bboxes_test)

  _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards, list_test, test_arr)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards,
                   list_train, train_arr)

if __name__ == '__main__':
  tf.app.run()
