'''
Example Pose Estimation Network with SE3 loss function (Training Script)
Dataset: KingsCollege
Network: Inception v1
Loss Function: Geomstats SE(3) Loss
'''

import argparse
import sys
import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'  # NOQA

import numpy as np
import geomstats.lie_group as lie_group
import tensorflow as tf

from geomstats.special_euclidean_group import SpecialEuclideanGroup
from tensorflow.contrib.slim.python.slim.nets import inception
from tqdm import tqdm


# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Train SE3 PoseNet Inception v1 Model.')
ARGPARSER.add_argument(
    '--batch_size', type=int, default=32,
    help='Batch size to train.')
ARGPARSER.add_argument(
    '--init_lr', type=float, default=1e-4,
    help='Initial Learning rate.')
ARGPARSER.add_argument(
    '--max_iter', type=int, default=200000,
    help='The number of iteration to train.')
ARGPARSER.add_argument(
    '--epsilon', type=float, default=np.finfo(np.float32).eps, # 1.1920929e-07
    help='Gradient Epsilon')
ARGPARSER.add_argument(
    '--snapshot', type=int, default=10000,
    help='Save model weights every X iterations')
ARGPARSER.add_argument(
    '--dataset', type=str, default='dataset_train.tfrecords',
    help='Training dataset')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='./model',
    help='The path to the model directory.')
ARGPARSER.add_argument(
    '--logs_path', type=str, default='./logs',
    help='The path to the logs directory.')
ARGPARSER.add_argument(
    '--cuda', type=str, default='0',
    help='Specify default GPU to use.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode.")


class PoseNetReader:

    def __init__(self, tfrecord_list):

        self.file_q = tf.train.string_input_producer(tfrecord_list)

    def read_and_decode(self, batch_size):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(self.file_q)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image':        tf.FixedLenFeature([], tf.string),
                'pose':         tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        pose = tf.decode_raw(features['pose'], tf.float32)

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
                                                          batch_size=batch_size,
                                                          capacity=1024,
                                                          num_threads=2,
                                                          min_after_dequeue=10)

        return image_batch , pose_batch


def main(args):

    SE3_GROUP = SpecialEuclideanGroup(3, epsilon=FLAGS.epsilon)
    metric = SE3_GROUP.left_canonical_metric

    reader_train = PoseNetReader([FLAGS.dataset])

    # Get Input Tensors
    image, y_true = reader_train.read_and_decode(FLAGS.batch_size)

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
        train_op = tf.train.AdamOptimizer(FLAGS.init_lr).minimize(loss)

    # Initialize the variables (i.e. assign their default value)
    print('Initalizing Variables...')
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a summary to monitor cost tensor
    tf.summary.scalar('loss', loss)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Main Training Routine
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init_op)

        # Start Queue Threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(FLAGS.logs_path, graph=tf.get_default_graph())

        saver = tf.train.Saver()
        # latest_checkpoint = tf.train.latest_checkpoint(_model_ckpt)
        # saver.restore(sess, latest_checkpoint)

        # Training cycle
        try:
            train_range = tqdm(range(FLAGS.max_iter))
            for i in train_range:

                _, _cost, summary = sess.run([train_op, loss, merged_summary_op])

                # Write logs at every iteration
                train_range.set_description('Training: (loss=%g)' % _cost)
                summary_writer.add_summary(summary, i)

                if i % FLAGS.snapshot == 0:
                    save_path = saver.save(sess, '{}/chkpt{}.ckpt'.format(FLAGS.model_dir, i))

        except KeyboardInterrupt:
            print('KeyboardInterrupt!')

        finally:
            print('Stopping Threads')
            coord.request_stop()
            coord.join(threads)
            print('Saving iter: ', i)
            save_path = saver.save(sess, FLAGS.model_dir + str(i) + '.ckpt')


if __name__ == '__main__':

    print('Training SE3 PoseNet Inception v1 Model.')
    FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
    print('FLAGS:', FLAGS)

    # Set verbosity
    if FLAGS.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda

    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED_ARGV)
