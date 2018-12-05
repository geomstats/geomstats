# -*- coding: utf-8 -*-
# ================================================================================
# Example Pose Estimation Network with SE3 loss function (Training Script)
# Dataset: KingsCollege
# Network: Inception v1
# Loss Function: Geomstats SE(3) Loss
# ================================================================================

import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'  # NOQA

import numpy as np
import geomstats.lie_group as lie_group
import tensorflow as tf

from geomstats.special_euclidean_group import SpecialEuclideanGroup
from tensorflow.contrib.slim.python.slim.nets import inception
from tqdm import tqdm

# ================================================================================
# Parameters
# ================================================================================

_batch_size     = 32
_init_lr        = 1e-4
_max_iter       = 200000
_snapshot       = 10000
_epsilon        = np.finfo(np.float32).eps # 1.1920929e-07

_logs_path      = 'logs/'
_ckpt_path      = 'model_ckpt/model_'

SE3_GROUP       = SpecialEuclideanGroup(3, epsilon=_epsilon)
metric          = SE3_GROUP.left_canonical_metric

# Datasets
filename_train  = ['dataset_train.tfrecords']
filename_test   = ['dataset_test.tfrecords']

# ================================================================================
# Reader Class
# ================================================================================

class PoseNetReader:

    def __init__(self, tfrecord_list):

        self.file_q = tf.train.string_input_producer(tfrecord_list)

    def read_and_decode(self):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(self.file_q)

        features = tf.parse_single_example(
            serialized_example,
            features={
                #'height':      tf.FixedLenFeature([], tf.int64),
                #'width':       tf.FixedLenFeature([], tf.int64),
                'image':        tf.FixedLenFeature([], tf.string),
                'pose':         tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        pose = tf.decode_raw(features['pose'], tf.float32)

        #height = tf.cast(features['height'], tf.int32)
        #width = tf.cast(features['width'], tf.int32)

        image = tf.reshape(image, (480, 270, 3))
        pose.set_shape((6))

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        # image = tf.image.resize_images(image, size=[224, 224])
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=224,
                                                       target_width=224)

        image_batch , pose_batch = tf.train.shuffle_batch([image, pose],
                                                          batch_size=_batch_size,
                                                          capacity=1024,
                                                          num_threads=2,
                                                          min_after_dequeue=10)

        return image_batch , pose_batch

# ================================================================================
# Network Definition
# ================================================================================

reader_train = PoseNetReader(filename_train)

# Get Input Tensors
image, y_true = reader_train.read_and_decode()

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
print('Making Model')
with tf.name_scope('Model'):
    py_x, _ = inception.inception_v1(tf.cast(image, tf.float32), num_classes=6)
    # tanh(pred_angle) required to prevent infinite spins on rotation axis
    y_pred = tf.concat((tf.nn.tanh(py_x[:, :3]), py_x[:, 3:]), axis=1)
    loss = tf.reduce_mean(lie_group.loss(y_pred, y_true, SE3_GROUP, metric))

print('Making Optimizer')
with tf.name_scope('Adam'):
    # Adam Optimizer
    train_op = tf.train.AdamOptimizer(_init_lr).minimize(loss)

# Initialize the variables (i.e. assign their default value)
print('Initalizing Variables...')
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar('loss', loss)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()


# ================================================================================
# Training Routine
# ================================================================================

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Start Queue Threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(_logs_path, graph=tf.get_default_graph())

    saver = tf.train.Saver()
    # latest_checkpoint = tf.train.latest_checkpoint(_model_ckpt)
    # saver.restore(sess, latest_checkpoint)

    # Training cycle
    try:
        train_range = tqdm(range(_max_iter))
        for i in train_range:

            _, _cost, summary = sess.run([train_op, loss, merged_summary_op])

            # Write logs at every iteration
            train_range.set_description('Training: (loss=%g)' % _cost)
            summary_writer.add_summary(summary, i)

            if i % _snapshot == 0:
                save_path = saver.save(sess, _ckpt_path + str(i) + '.ckpt')

    except KeyboardInterrupt:
        print('KeyboardInterrupt!')

    finally:
        print('Stopping Threads')
        coord.request_stop()
        coord.join(threads)
        print('Saving iter: ', i)
        save_path = saver.save(sess, _ckpt_path + str(i) + '.ckpt')
