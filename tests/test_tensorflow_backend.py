"""
Unit tests for tensorflow backend.
"""

import importlib
import os
import tensorflow as tf

import geomstats.backend as gs
import tests.helper as helper

from geomstats.hypersphere import Hypersphere


class TestHypersphereOnTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 4
        self.space = Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10
        self.depth = 3

    @classmethod
    def setUpClass(cls):
        tf.enable_eager_execution()
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_random_uniform_and_belongs(self):
        pass
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        point = self.space.random_uniform()
        with self.test_session():
            self.assertTrue(gs.eval(self.space.belongs(point)[0, 0]))

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        with self.test_session():
            base_point = gs.array([16., -2., -2.5, 84., 3.])
            base_point = base_point / gs.linalg.norm(base_point)
            vector = gs.array([9., 0., -1., -2., 1.])
            tangent_vec = self.space.projection_to_tangent_space(
                                                      vector=vector,
                                                      base_point=base_point)

            exp = self.metric.exp(tangent_vec=tangent_vec,
                                  base_point=base_point)
            result = self.metric.dist(base_point, exp)
            expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
            expected = helper.to_scalar(expected)
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_exp_and_dist_and_projection_to_tangent_space_vec(self):
        with self.test_session():

            base_point = gs.array([[16., -2., -2.5, 84., 3.],
                                   [16., -2., -2.5, 84., 3.]])

            base_single_point = gs.array([16., -2., -2.5, 84., 3.])
            scalar_norm = gs.linalg.norm(base_single_point)

            base_point = base_point / scalar_norm
            vector = gs.array([[9., 0., -1., -2., 1.], [9., 0., -1., -2., 1]])

            tangent_vec = self.space.projection_to_tangent_space(
                    vector=vector,
                    base_point=base_point)

            exp = self.metric.exp(tangent_vec=tangent_vec,
                                  base_point=base_point)

            result = self.metric.dist(base_point, exp)
            expected = gs.linalg.norm(tangent_vec, axis=-1) % (2 * gs.pi)

            expected = helper.to_scalar(expected)
            self.assertAllClose(gs.eval(result), gs.eval(expected))


if __name__ == '__main__':
    tf.test.main()
