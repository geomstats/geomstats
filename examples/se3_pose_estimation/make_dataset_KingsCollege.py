# -*- coding: utf-8 -*-
# ================================================================================
# Original
# File: create_posenet_lmdb_dataset.py
# Link: https://git.io/fpPuw
# Author: Alex Kendall <https://alexgkendall.com>
#
# Modified
# File: make_dataset_Kingscollege.py
# Author: Benjamin Hou <bh1511@imperial.ac.ukm>
#
# Download KingsCollege Dataset:
# http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset
# ================================================================================

import imageio
import random

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


###############################################################################
# Tensorflow feature wrapper

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


###############################################################################
# Dataset Locations

directory   = '/vol/medic01/users/bh1511/4_CVPR2019/data_raw/KingsCollege/'
dataset     = 'dataset_test.txt'
out_file    = 'dataset_test.tfrecords'


###############################################################################
# Process Lables

poses   = []
images  = []

with open(directory+dataset) as f:
    next(f)  # skip the 3 header lines
    next(f)
    next(f)
    for line in f:
        fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
        p0 = float(p0)
        p1 = float(p1)
        p2 = float(p2)
        p3 = float(p3)
        p4 = float(p4)
        p5 = float(p5)
        p6 = float(p6)
        poses.append((p0, p1, p2, p3, p4, p5, p6))
        images.append(directory+fname)

r = list(range(len(images)))

random.shuffle(r)
random.shuffle(r)
random.shuffle(r)


###############################################################################
# Create Database

print('Creating PoseNet Dataset.')

SO3_GROUP   = SpecialOrthogonalGroup(3)
writer      = tf.python_io.TFRecordWriter(out_file)

for i in tqdm(r):

    pose_q  = np.array(poses[i][3:7])
    pose_x  = np.array(poses[i][0:3])

    rot_vec = SO3_GROUP.rotation_vector_from_quaternion(pose_q)[0]
    pose    = np.concatenate((rot_vec, pose_x), axis=0)

    X = imageio.imread(images[i])
    X = X[::4, ::4, :]
    #X = exposure.equalize_hist(X)

    img_raw     = X.tostring() #.astype('float32').tostring()
    pose_raw    = pose.astype('float32').tostring()
    pose_q_raw  = pose_q.astype('float32').tostring()
    pose_x_raw  = pose_x.astype('float32').tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height':   _int64_feature(X.shape[0]),
        'width':    _int64_feature(X.shape[1]),
        'channel':  _int64_feature(X.shape[2]),
        'image':    _bytes_feature(img_raw),
        'pose':     _bytes_feature(pose_raw),
        'pose_q':   _bytes_feature(pose_q_raw),
        'pose_x':   _bytes_feature(pose_x_raw)}))

    writer.write(example.SerializeToString())

writer.close()


print('\n', 'Creating Dataset Success.')
