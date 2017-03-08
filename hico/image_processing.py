from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 1,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('num_relations', 600,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def inputs(dataset, batch_size=None, image_size=None, num_preprocess_threads=None):
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels = batch_inputs(
        dataset, batch_size, image_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=1)

  return images, labels


def distorted_inputs(dataset, batch_size=None, image_size=None, num_preprocess_threads=None):
  if not batch_size:
    batch_size = FLAGS.batch_size
  if not image_size:
    image_size = 64

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels, filenames = batch_inputs(
        dataset, batch_size, image_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=FLAGS.num_readers)
  return images, labels, filenames


def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  with tf.op_scope([image], scope, 'distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_image(image, height, width, thread_id=0, scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  with tf.op_scope([image, height, width], scope, 'distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # Crop the image to the specified bounding box.
    #distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = image
    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    #resize_method = thread_id % 4
    resize_method = 0
    distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                             method=resize_method)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    #distorted_image.set_shape([height, width, 3])

    if not thread_id:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))


    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    #distorted_image = distort_color(distorted_image, thread_id)


    if not thread_id:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    return distorted_image


def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.op_scope([image, height, width], scope, 'eval_image'):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image


def image_preprocessing(image_buffer, image_size, train, thread_id=0):
  #if bbox is None:
  #  raise ValueError('Please supply a bounding box.')

  image = decode_jpeg(image_buffer)
  height = image_size
  width = image_size

  if train:
    image = distort_image(image, height, width, thread_id)
  else:
    image = eval_image(image, height, width)

  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _bytes_feature(label.tostring()),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))

    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/height': tf.FixedLenFeature([], dtype=tf.int64,
                                        default_value=256),
      'image/width': tf.FixedLenFeature([], dtype=tf.int64,
                                        default_value=256),
  }
  #sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  ## Sparse features in Example proto.
  #feature_map.update(
  #    {k: sparse_float32 for k in ['image/object/bbox/xmin',
  #                                 'image/object/bbox/ymin',
  #                                 'image/object/bbox/xmax',
  #                                 'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  #label = tf.cast(features['image/class/label'], dtype=tf.int32)
  #filename = tf.cast(features['image/filename'], dtype=tf.string)

  #xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  #ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  #xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  #ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  ## Note that we impose an ordering of (y, x) just to make life difficult.
  #bbox = tf.concat(0, [ymin, xmin, ymax, xmax])

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  #bbox = tf.expand_dims(bbox, 0)
  #bbox = tf.transpose(bbox, [0, 2, 1])
  return features['image/encoded'], features['image/class/label'], features['image/filename']


def batch_inputs(dataset, batch_size, image_size, train, num_preprocess_threads=None,
                 num_readers=1):
  with tf.name_scope('batch_processing'):
    data_files = dataset.data_files()
    if data_files is None:
      raise ValueError('No data files found for this dataset')
    #else:
    #  print(data_files)

    # Create filename_queue
    if train:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=True,
                                                      capacity=16)
    else:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    if num_preprocess_threads is None:
      num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
      raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
      num_readers = FLAGS.num_readers

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if train:
      examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])
    else:
      examples_queue = tf.FIFOQueue(
          capacity=examples_per_shard + 3 * batch_size,
          dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        reader = dataset.reader()
        _, value = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
    else:
      reader = dataset.reader()
      _, example_serialized = reader.read(filename_queue)

    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      #image_buffer, label_index, bbox, _ = parse_example_proto(
      #    example_serialized)
      image_buffer, label_buffer, filename = parse_example_proto(
          example_serialized)
      image = image_preprocessing(image_buffer, image_size, train, thread_id)
      label_index = tf.decode_raw(label_buffer, out_type = tf.int32)
      label_index = tf.reshape(label_index, shape = [FLAGS.num_relations]) 
      images_and_labels.append([image, label_index, filename])

    images, label_index_batch, filenames = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size)

    # Reshape images into these desired dimensions.
    height = image_size
    width = image_size
    depth = 3

    #images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, tf.reshape(label_index_batch, [batch_size, FLAGS.num_relations]), tf.reshape(filenames, [batch_size])
