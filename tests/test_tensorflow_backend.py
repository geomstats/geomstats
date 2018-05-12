"""Unit tests for hypersphere module."""
import numpy as np
import os
os.environ['GEOMSTATS_BACKEND'] = 'numpy'
from geomstats import hypersphere

import tests.helper as helper

import geomstats.backend as gs
import importlib

import tensorflow as tf



class TestHypersphereMethods(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 4
        self.space = hypersphere.Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10
        self.depth = 3

    @classmethod
    def etUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)
        importlib.reload(hypersphere)

    @classmethod
    def tearDownClass(cls):
        os.unsetenv('GEOMSTATS_BACKEND')
        importlib.reload(gs)

    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        point = self.space.random_uniform()
        with self.test_session():
            print(self.space.random_uniform(n_samples=1, depth=3))
            self.assertTrue(self.space.belongs(point)) #.eval())

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        with self.test_session():
            base_point = gs.array([16., -2., -2.5, 84., 3.])
            base_point = base_point / gs.linalg.norm(base_point)

            vector = gs.array([9., 0., -1., -2., 1.])
            tangent_vec = self.space.projection_to_tangent_space(
                                                          vector=vector,
                                                          base_point=base_point)
            print(tangent_vec)
            exp = self.metric.exp(tangent_vec=tangent_vec,
                                  base_point=base_point)
            print('expo: %s' % exp)
            result = self.metric.dist(base_point, exp)
            expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
            expected = helper.to_scalar(expected)
            print(expected)
            print(result)
            self.assertAllClose(result, expected) #.eval(), expected.eval())

    def test_exp_and_dist_and_projection_to_tangent_space_vec(self):
        with self.test_session():
            base_point = gs.array([[16., -2., -2.5, 84., 3.],
                                   [16., -2., -2.5, 84., 3.]])
            base_point = np.expand_dims(base_point, 0)
            base_point = base_point / gs.linalg.norm(base_point)

            vector = gs.array([[9., 0., -1., -2., 1.], [9., 0., -1., -2., 1]])
            vector = np.expand_dims(vector, 0)
            tangent_vec = self.space.projection_to_tangent_space(
                                                          vector=vector,
                                                          base_point=base_point)
            print(tangent_vec)
            exp = self.metric.exp(tangent_vec=tangent_vec,
                                  base_point=base_point)
            print('expo2 %s' % exp)
            result = self.metric.dist(base_point, exp)
            expected = gs.linalg.norm(tangent_vec, axis=-1) % (2 * gs.pi)
            #print('result:%s' % result)
            #print('expected %s' % expected)
            expected = helper.to_scalar(expected)
            self.assertAllClose(result, expected) # .eval(), expected.eval())

def to_vector(expected):
    expected = gs.to_ndarray(expected, to_ndim=2)
    expected = gs.to_ndarray(expected, to_ndim=3, axis=1)
    return expected

if __name__ == '__main__':
    tf.test.main()
