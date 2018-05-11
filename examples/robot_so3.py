"""
Generate a geodesic of SO(3) equipped
with its left-invariant canonical METRIC
for trajectory generation for a robotic manipulator
and sends the trajectory through a redis server at a selected rate
"""

import time

from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

import numpy as np

import redis

SO3_GROUP = SpecialOrthogonalGroup(n=3)
METRIC = SO3_GROUP.bi_invariant_metric

redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)
DESIRED_ORIENTATION_KEY = "geomstats_examples::desired_orientation"
DESIRED_POSITION_KEY = "geomstats_examples::desired_position"

trajectory_time_seconds = 5.0
loop_frequency_Hz = 100.0


def decode_vector_redis(redis_key):
    """
    reads a the value corresponding to 'redis_key'
    from the redis server and returns it as a np 1D array
    """
    return np.array(str(redis_server.get(redis_key)).split('\'')[1].split(' '))


def decode_matrix_redis(redis_key):
    """
    reads a the value corresponding to 'redis_key'
    from the redis server and returns it as a 2D array
    """
    lines = str(redis_server.get(redis_key)).split('\'')[1].split('; ')
    matrix = np.array([x.split(' ') for x in lines])

    return matrix.astype(np.float)


def encode_matrix_redis(redis_key, mat):
    """
    writes a np 2D array 'mat' to the redis server
    using the key 'redis_key'
    """
    s = ''

    for j in range(mat.shape[1]):
        s += str(mat[0][j])
        s += ' '
    s = s[0:-1]
    s += '; '

    for i in range(1, mat.shape[0]):
        for j in range(mat.shape[1]):
            s += str(mat[i][j])
            s += ' '
        s = s[0:-1]
        s += '; '
    s = s[0:-2]

    redis_server.set(redis_key, s)


def main():

    initial_orientation = decode_matrix_redis(DESIRED_ORIENTATION_KEY)
    initial_point = SO3_GROUP.rotation_vector_from_matrix(initial_orientation)

    final_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    final_point = SO3_GROUP.rotation_vector_from_matrix(final_orientation)

    geodesic = METRIC.geodesic(initial_point=initial_point,
                               end_point=final_point)

    n_steps = int(trajectory_time_seconds * loop_frequency_Hz)
    t = np.linspace(0, 1, n_steps)
    current_step = 0

    points = geodesic(t)

    period = 1.0 / loop_frequency_Hz
    t_init = time.time()
    t = t_init

    while(current_step < n_steps):
        t += period

        current_point = points[current_step]
        rot_desired = SO3_GROUP.matrix_from_rotation_vector(current_point)[0]
        encode_matrix_redis(DESIRED_ORIENTATION_KEY, rot_desired)

        current_step = current_step + 1
        time.sleep(max(0.0, t - time.time()))

    elapsed_time = time.time() - t_init
    print("Elapsed time : ", elapsed_time, " seconds")
    print("Loop cycles  : ", current_step)
    print("Frequency    : ", current_step / elapsed_time, " Hz")

if __name__ == "__main__":
    main()
