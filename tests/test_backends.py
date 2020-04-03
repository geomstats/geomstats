"""
Unit tests for backends.

The functions are tested in order to match numpy's results and API.
In exceptional cases, numpy's results or API may not be followed.
"""

import warnings

import numpy as _np
import scipy.linalg

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestBackends(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        self.so3_group = SpecialOrthogonal(n=3)
        self.n_samples = 2

    def test_matmul(self):
        mat_a = [[2., 0., 0.],
                 [0., 3., 0.],
                 [7., 0., 4.]]
        mat_b = [[1., 0., 2.],
                 [0., 3., 0.],
                 [0., 0., 1.]]
        gs_mat_a = gs.array(mat_a)
        gs_mat_b = gs.array(mat_b)
        np_mat_a = _np.array(mat_a)
        np_mat_b = _np.array(mat_b)

        gs_result = gs.matmul(gs_mat_a, gs_mat_b)
        np_result = _np.matmul(np_mat_a, np_mat_b)

        self.assertAllCloseToNp(gs_result, np_result)

    @geomstats.tests.np_and_tf_only
    def test_matmul_vectorization(self):
        mat_a = [[2., 0., 0.],
                 [0., 3., 0.],
                 [7., 0., 4.]]
        mat_b = [[1., 0., 2.],
                 [0., 3., 0.],
                 [0., 0., 1.]]
        mat_c = [[1., 4., 2.],
                 [4., 3., 4.],
                 [0., 0., 4.]]
        gs_mat_a = gs.array(mat_a)
        gs_mat_b = gs.array(mat_b)
        gs_mat_c = gs.array(mat_c)
        np_mat_a = _np.array(mat_a)
        np_mat_b = _np.array(mat_b)
        np_mat_c = _np.array(mat_c)

        gs_result = gs.matmul(gs_mat_a, [gs_mat_b, gs_mat_c])
        np_result = _np.matmul(np_mat_a, [np_mat_b, np_mat_c])

        self.assertAllCloseToNp(gs_result, np_result)

    @geomstats.tests.np_and_tf_only
    def test_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.logm(point)
        expected = gs.array([[0.693147180, 0., 0.],
                             [0., 1.098612288, 0.],
                             [0., 0., 1.38629436]])
        self.assertAllClose(result, expected)

        np_point = _np.array(
            [[2., 0., 0.],
             [0., 3., 0.],
             [0., 0., 4.]])
        scipy_result = scipy.linalg.logm(np_point)
        self.assertAllCloseToNp(result, scipy_result)

    @geomstats.tests.np_and_tf_only
    def test_expm_and_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point
        self.assertAllClose(result, expected)

        np_point = _np.array(
            [[2., 0., 0.],
             [0., 3., 0.],
             [0., 0., 4.]])
        scipy_result = scipy.linalg.expm(scipy.linalg.logm(np_point))
        self.assertAllCloseToNp(result, scipy_result)

    @geomstats.tests.np_only
    def test_expm_vectorization(self):
        # Note: scipy.linalg.expm is not vectorized
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[7.38905609, 0., 0.],
                              [0., 20.0855369, 0.],
                              [0., 0., 54.5981500]],
                             [[2.718281828, 0., 0.],
                              [0., 148.413159, 0.],
                              [0., 0., 403.42879349]]])

        result = gs.linalg.expm(point)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_logm_vectorization_diagonal(self):
        # Note: scipy.linalg.expm is not vectorized
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[0.693147180, 0., 0.],
                              [0., 1.09861228866, 0.],
                              [0., 0., 1.38629436]],
                             [[0., 0., 0.],
                              [0., 1.609437912, 0.],
                              [0., 0., 1.79175946]]])

        result = gs.linalg.logm(point)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_expm_and_logm_vectorization_random_rotation(self):
        point = self.so3_group.random_uniform(self.n_samples)
        point = self.so3_group.matrix_from_rotation_vector(point)

        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_expm_and_logm_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_powerm_diagonal(self):
        power = .5
        point = gs.array([[1., 0., 0.],
                          [0., 4., 0.],
                          [0., 0., 9.]])
        result = gs.linalg.powerm(point, power)
        expected = gs.array([[1., 0., 0.],
                             [0., 2., 0.],
                             [0., 0., 3.]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_powerm(self):
        power = 2.4
        point = gs.array([[1., 0., 0.],
                          [0., 2.5, 1.5],
                          [0., 1.5, 2.5]])
        result = gs.linalg.powerm(point, power)
        result = gs.linalg.powerm(result, 1 / power)
        expected = point

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_powerm_vectorization(self):
        power = 2.4
        points = gs.array([[[1., 0., 0.],
                            [0., 4., 0.],
                            [0., 0., 9.]],
                           [[1., 0., 0.],
                            [0., 2.5, 1.5],
                            [0., 1.5, 2.5]]])
        result = gs.linalg.powerm(points, power)
        result = gs.linalg.powerm(result, 1. / power)
        expected = points

        self.assertAllClose(result, expected)

    @geomstats.tests.pytorch_only
    def test_sampling_choice(self):
        res = gs.random.choice(10, (5, 1, 3))
        self.assertAllClose(res.shape, [5, 1, 3])

    @geomstats.tests.tf_only
    def test_vstack(self):
        import tensorflow as tf
        tensor_1 = tf.convert_to_tensor([[1., 2., 3.], [4., 5., 6.]])
        tensor_2 = tf.convert_to_tensor([[7., 8., 9.]])

        result = gs.vstack([tensor_1, tensor_2])
        expected = tf.convert_to_tensor([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
        self.assertAllClose(result, expected)

    @geomstats.tests.tf_only
    def test_tensor_addition(self):
        tensor_1 = gs.ones((1, 1))
        tensor_2 = gs.ones((0, 1))

        tensor_1 + tensor_2

    @geomstats.tests.pytorch_only
    def test_cumsum(self):
        result = gs.cumsum(gs.arange(10))
        expected = gs.array(([0, 1, 3, 6, 10, 15, 21, 28, 36, 45]))
        self.assertAllClose(result, expected)

        result = gs.cumsum(gs.arange(10).reshape(2, 5), axis=1)
        expected = gs.array(([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]]))
        self.assertAllClose(result, expected)

    def test_einsum(self):
        np_array_1 = _np.array([[1, 4]])
        np_array_2 = _np.array([[2, 3]])
        array_1 = gs.array([[1, 4]])
        array_2 = gs.array([[2, 3]])

        np_result = _np.einsum('...i,...i->...', np_array_1, np_array_2)
        gs_result = gs.einsum('...i,...i->...', array_1, array_2)

        self.assertAllCloseToNp(gs_result, np_result)

        np_array_1 = _np.array([[1, 4], [-1, 5]])
        np_array_2 = _np.array([[2, 3]])
        array_1 = gs.array([[1, 4], [-1, 5]])
        array_2 = gs.array([[2, 3]])

        np_result = _np.einsum('...i,...i->...', np_array_1, np_array_2)
        gs_result = gs.einsum('...i,...i->...', array_1, array_2)

        self.assertAllCloseToNp(gs_result, np_result)

        np_array_1 = _np.array([[1, 4]])
        np_array_2 = _np.array([[2, 3], [5, 6]])
        array_1 = gs.array([[1, 4]])
        array_2 = gs.array([[2, 3], [5, 6]])

        np_result = _np.einsum('...i,...i->...', np_array_1, np_array_2)
        gs_result = gs.einsum('...i,...i->...', array_1, array_2)

        self.assertAllCloseToNp(gs_result, np_result)

    def test_assignment(self):
        np_array_1 = _np.ones(3)
        gs_array_1 = gs.ones_like(gs.array(np_array_1))

        np_array_1[2] = 1.5
        gs_result = gs.assignment(gs_array_1, 1.5, 2)
        self.assertAllCloseToNp(gs_result, np_array_1)

        np_array_2 = _np.zeros((3, 2))
        gs_array_2 = gs.zeros_like(gs.array(np_array_2))

        np_array_2[0, :] = 1
        gs_result = gs.assignment(gs_array_2, 1, 0, axis=0)
        self.assertAllCloseToNp(gs_result, np_array_2)

        np_array_3 = _np.zeros((3, 3))
        gs_array_3 = gs.zeros_like(gs.array(np_array_3))

        np_array_3[0, 1] = 1
        gs_result = gs.assignment(gs_array_3, 1, (0, 1))
        self.assertAllCloseToNp(gs_result, np_array_3)

        np_array_4 = _np.zeros((3, 3, 2))
        gs_array_4 = gs.zeros_like(gs.array(np_array_4))

        np_array_4[0, :, 1] = 1
        gs_result = gs.assignment(gs_array_4, 1, (0, 1), axis=1)
        self.assertAllCloseToNp(gs_result, np_array_4)


    def test_assignment_by_sum(self):
        np_array_1 = _np.ones(3)
        gs_array_1 = gs.ones_like(gs.array(np_array_1))

        np_array_1[2] += 1.5
        gs_result = gs.assignment_by_sum(gs_array_1, 1.5, 2)
        self.assertAllCloseToNp(gs_result, np_array_1)

        np_array_2 = _np.zeros((3, 2))
        gs_array_2 = gs.zeros_like(gs.array(np_array_2))

        np_array_2[0, :] += 1
        gs_result = gs.assignment_by_sum(gs_array_2, 1, 0, axis=0)
        self.assertAllCloseToNp(gs_result, np_array_2)

        np_array_3 = _np.zeros((3, 3))
        gs_array_3 = gs.zeros_like(gs.array(np_array_3))

        np_array_3[0, 1] += 1
        gs_result = gs.assignment_by_sum(gs_array_3, 1, (0, 1))
        self.assertAllCloseToNp(gs_result, np_array_3)

        np_array_4 = _np.zeros((3, 3, 2))
        gs_array_4 = gs.zeros_like(gs.array(np_array_4))

        np_array_4[0, :, 1] += 1
        gs_result = gs.assignment_by_sum(gs_array_4, 1, (0, 1), axis=1)
        self.assertAllCloseToNp(gs_result, np_array_4)



