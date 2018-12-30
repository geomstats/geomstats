'''
Example Pose Estimation Network with SE3 loss function (Inference Script)
Dataset: KingsCollege
Network: Inception v1
Loss Function: Geomstats SE(3) Loss
'''

import argparse
import sys
import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'  # NOQA

import geomstats.lie_group as lie_group
import tensorflow as tf

from geomstats.special_euclidean_group import SpecialEuclideanGroup
from tensorflow.contrib.slim.python.slim.nets import inception


# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Test SE3 PoseNet Inception v1 Model.')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='./model',
    help='The path to the model directory.')
ARGPARSER.add_argument(
    '--dataset', type=str, default='dataset_test.tfrecords',
    help='The path to the TFRecords dataset.')
ARGPARSER.add_argument(
    '--cuda', type=str, default='0',
    help='Specify default GPU to use.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode.")


class PoseNetReader:

    def __init__(self, tfrecord_list):

        self.file_q = tf.train.string_input_producer(
            tfrecord_list, num_epochs=1)

    def read_and_decode(self):
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

        image = tf.reshape(image, (1, 480, 270, 3))
        pose.set_shape((6))

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        # image = tf.image.resize_images(image, size=[224, 224])
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=224,
                                                       target_width=224)

        return image, pose


def main(args):

    SE3_GROUP = SpecialEuclideanGroup(3)
    metric = SE3_GROUP.left_canonical_metric

    reader_train = PoseNetReader([FLAGS.dataset])

    # Get Input Tensors
    image, y_true = reader_train.read_and_decode()

    # Construct model and encapsulating all ops into scopes, making
    # Tensorboard's Graph visualization more convenient
    print('Making Model')
    with tf.name_scope('Model'):
        py_x, _ = inception.inception_v1(tf.cast(image, tf.float32),
                                         num_classes=6,
                                         is_training=False)
        # tanh(pred_angle) required to prevent infinite spins on rotation axis
        y_pred = tf.concat((tf.nn.tanh(py_x[:, :3]), py_x[:, 3:]), axis=1)
        loss = tf.reduce_mean(
            lie_group.loss(y_pred, y_true, SE3_GROUP, metric))

    print('Initalizing Variables...')
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Main Testing Routine
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init_op)

        # Start Queue Threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Load saved weights
        print('Loading Trained Weights')
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        saver.restore(sess, latest_checkpoint)

        i = 0

        # Inference cycle
        try:
            while True:
                _y_pred, _y_true, _loss = sess.run([y_pred, y_true, loss])
                print('Iteration:', i, 'loss:', _loss)
                print('_y_pred:', _y_pred)
                print('_y_true:', _y_true)
                print('\n')
                i = i + 1

        except tf.errors.OutOfRangeError:
            print('End of Testing Data')

        except KeyboardInterrupt:
            print('KeyboardInterrupt!')

        finally:
            print('Stopping Threads')
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    print('Testing SE3 PoseNet Inception v1 Model.')
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
