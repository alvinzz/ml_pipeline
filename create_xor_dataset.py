from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow==2.0.0-beta1
import tensorflow as tf
import numpy as np

# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#write
x = np.array([[0,0], [0,1], [1,0], [1,1]]).astype(np.float32)
y = np.array([0, 1, 1, 0]).astype(np.float32)

def gen_example(x, y):
  feature = {
      'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.tolist())),
      'y': tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = 'xor.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for (_x, _y) in zip(x, y):
        example = gen_example(_x, _y)
        writer.write(example.SerializeToString())


#read
@tf.function
def read():
    with tf.device('/cpu:0'):
        raw_image_dataset = tf.data.TFRecordDataset('xor.tfrecords')

        # Create a dictionary describing the features.
        feature_description = {
            'x': tf.io.FixedLenFeature([2], tf.float32),
            'y': tf.io.FixedLenFeature([], tf.float32),
        }
        def example_parser(example):
          return tf.io.parse_single_example(example, feature_description)

        dataset = tf.data.TFRecordDataset(filenames=['xor.tfrecords'])
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=4, seed=0)
        dataset = dataset.map(example_parser)
        dataset = dataset.batch(4)
        dataset = dataset.prefetch(1)
        batch = next(iter(dataset))

        for _ in range(100):
            print(batch)

read()

# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

# raw_example = next(iter(raw_image_dataset))
# parsed = _parse_image_function(raw_example)

# print(parsed)
