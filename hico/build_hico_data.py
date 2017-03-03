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

tf.app.flags.DEFINE_string('train_directory', '/media/hao/SeagateDisk2/dataset/hico_20150920/images/train2015/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/media/hao/SeagateDisk2/dataset/hico_20150920/images/test2015/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/media/hao/SeagateDisk2/dataset/hico_20150920/tfrecord',
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


def _convert_to_example(filename, image_buffer, label,
                        height, width):

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _bytes_feature(label.tostring()),
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
                                labels, num_shards):
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
      label = labels[:, i]
      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, label, 
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


def _process_image_files(name, filenames, list_obj, list_action, list_relation, 
                        label, num_shards):

  assert len(filenames) == label.shape[1]

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
    args = (coder, thread_index, ranges, name, filenames, label, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _process_dataset(name, directory, num_shards, list_image, list_obj, 
                    list_action, list_relation, label):
  filenames = []
  for i in list_image:
    path = os.path.join(directory, i)
    filenames.append(path)
  _process_image_files(name, filenames, list_obj, list_action, list_relation,
                       label, num_shards)

def _read_annotations():
  anno_file = '/media/hao/WDdisk/relation2image/data/hico_benchmark/hico_20150920/anno.mat'
  mat = scipy.io.loadmat(anno_file) 
  list_train_raw, list_test_raw = mat['list_train'], mat['list_test']
  list_train, list_test = [], []
  for im_name in list_train_raw:
    list_train.append(im_name[0][0])
  for im_name in list_test_raw:
    list_test.append(im_name[0][0])
  
  # read obj list
  list_obj = []
  list_obj_file = '/media/hao/WDdisk/relation2image/data/hico_benchmark/hico_20150920/hico_list_obj.txt'
  with open(list_obj_file, 'r') as f:
    line_cnt = 0
    for line in f.readlines():
      line_cnt += 1
      if line_cnt >= 3:
        index, name = line.rstrip('\n').split()
        list_obj.append(name)

  # read action list
  list_action = []
  list_action_file = '/media/hao/WDdisk/relation2image/data/hico_benchmark/hico_20150920/hico_list_vb.txt'
  with open(list_action_file, 'r') as f:
    line_cnt = 0
    for line in f.readlines():
      line_cnt += 1
      if line_cnt >= 3:
        index, name = line.rstrip('\n').split()
        list_action.append(name)

  # read relation list
  relations_raw = mat['list_action']
  list_relation = []
  for r in relations_raw:
    obj = r[0][0][0]
    obj_index = list_obj.index(obj)
    act = r[0][1][0]
    act_index = list_action.index(act)
    relation = [obj, act, obj_index, act_index]
    list_relation.append(relation)

  # read image labels -- their relations
  train_label_raw, test_label_raw = mat['anno_train'], mat['anno_test']
  train_label, test_label = np.zeros(train_label_raw.shape, dtype = np.int32), np.zeros(test_label_raw.shape, dtype = np.int32)
  for row in range(0, train_label_raw.shape[0]):
    for col in range(0, train_label_raw.shape[1]):
      entry = train_label_raw[row, col]
      if entry == 1.0:
        train_label[row, col] = 1
      elif entry == 0:
        train_label[row, col] = 0
      elif entry == -1.0:
        train_label[row, col] = -1
      else:
        train_label[row, col] = -2
  for row in range(0, test_label_raw.shape[0]):
    for col in range(0, test_label_raw.shape[1]):
      entry = test_label_raw[row, col]
      if entry == 1.0:
        test_label[row, col] = 1
      elif entry == 0:
        test_label[row, col] = 0
      elif entry == -1.0:
        test_label[row, col] = -1
      else:
        test_label[row, col] = -2
  
  # random shuffle the data
  arr = np.arange(train_label.shape[1])
  np.random.shuffle(arr)
  list_train_shuf = []
  train_label_shuf = np.zeros_like(train_label)
  for i in range(arr.shape[0]):
    list_train_shuf.append(list_train[arr[i]]) 
    train_label_shuf[:, i] = train_label[:, arr[i]]
  
  arr = np.arange(test_label.shape[1])
  np.random.shuffle(arr)
  list_test_shuf = []
  test_label_shuf = np.zeros_like(test_label)
  for i in range(arr.shape[0]):
    list_test_shuf.append(list_test[arr[i]]) 
    test_label_shuf[:, i] = test_label[:, arr[i]]
  return list_train_shuf, list_test_shuf, list_obj, list_action, list_relation, train_label_shuf, test_label_shuf

def _from_unicode_to_ascii(list_obj, list_action, list_train, list_test, list_relation):
  list_obj = [n.encode("ascii", "ignore") for n in list_obj]
  list_action = [n.encode("ascii", "ignore") for n in list_action]
  list_train = [n.encode("ascii", "ignore") for n in list_train]
  list_test= [n.encode("ascii", "ignore") for n in list_test]
  for i in list_relation:
    i[0] = i[0].encode("ascii", "ignore")
    i[1] = i[1].encode("ascii", "ignore")
  return list_obj, list_action, list_train, list_test, list_relation

def _save_meta(list_obj, list_action, list_train, list_test, list_relation):
  # first, save the metadata into hdf5
  meta = h5py.File(os.path.join('/media/hao/WDdisk/relation2image/data/hico_benchmark/hico_20150920/tfrecord', 'meta.h5'), 'w')
  meta.create_dataset('list_obj', data = list_obj)
  meta.create_dataset('list_action', data = list_action)
  meta.create_dataset('list_relation', data = list_relation)
  meta.create_dataset('list_train', data = list_train)
  meta.create_dataset('list_test', data = list_test)
  meta.close()

def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  list_train, list_test, list_obj, list_action, list_relation, train_label, test_label = _read_annotations()

  # Run it!
  list_obj, list_action, list_train, list_test, list_relation = _from_unicode_to_ascii(list_obj, list_action, list_train, list_test, list_relation)
  _save_meta(list_obj, list_action, list_train, list_test, list_relation)
  _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards, list_test, list_obj, list_action, list_relation, test_label)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards,
                   list_train, list_obj, list_action, list_relation, train_label)

if __name__ == '__main__':
  tf.app.run()
