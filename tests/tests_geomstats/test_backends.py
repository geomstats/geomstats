"""Unit tests for backends.

The functions are tested in order to match numpy's results and API.
In exceptional cases, numpy's results or API may not be followed.
"""

import warnings

import numpy as _np
import scipy.linalg
import torch

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestBackends(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        self.so3_group = SpecialOrthogonal(n=3)
        self.n_samples = 2

    def test_array(self):
        gs_mat = gs.array(1.5)
        np_mat = _np.array(1.5)
        self.assertAllCloseToNp(gs_mat, np_mat)

        gs_mat = gs.array([gs.ones(3), gs.ones(3)])
        np_mat = _np.array([_np.ones(3), _np.ones(3)])
        self.assertAllCloseToNp(gs_mat, np_mat)

        gs_mat = gs.array([gs.ones(3), gs.ones(3)], dtype=gs.float64)
        np_mat = _np.array([_np.ones(3), _np.ones(3)], dtype=_np.float64)
        self.assertTrue(gs_mat.dtype == gs.float64)
        self.assertAllCloseToNp(gs_mat, np_mat)

        gs_mat = gs.array([[gs.ones(3)], [gs.ones(3)]], dtype=gs.uint8)
        np_mat = _np.array([[_np.ones(3)], [_np.ones(3)]], dtype=_np.uint8)
        self.assertTrue(gs_mat.dtype == gs.uint8)
        self.assertAllCloseToNp(gs_mat, np_mat)

        gs_mat = gs.array([gs.ones(3), [0, 0, 0]], dtype=gs.int32)
        np_mat = _np.array([_np.ones(3), [0, 0, 0]], dtype=_np.int32)
        self.assertTrue(gs_mat.dtype == gs.int32)
        self.assertAllCloseToNp(gs_mat, np_mat)

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

    def test_cumprod(self):
        result = gs.cumprod(gs.arange(1, 10))
        expected = gs.array(([1, 2, 6, 24, 120, 720, 5040, 40320, 362880]))
        self.assertAllClose(result, expected)

        result = gs.reshape(gs.arange(1, 11), (2, 5))
        result = gs.cumprod(result, axis=1)
        expected = gs.array(([[1, 2, 6, 24, 120], [6, 42, 336, 3024, 30240]]))
        self.assertAllClose(result, expected)

    @geomstats.tests.pytorch_only
    def test_cumsum(self):
        result = gs.cumsum(gs.arange(10))
        expected = gs.array(([0, 1, 3, 6, 10, 15, 21, 28, 36, 45]))
        self.assertAllClose(result, expected)

        result = gs.cumsum(gs.arange(10).reshape(2, 5), axis=1)
        expected = gs.array(([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]]))
        self.assertAllClose(result, expected)

    def test_array_from_sparse(self):
        expected = gs.array([[0, 1, 0], [0, 0, 2]])
        result = gs.array_from_sparse([(0, 1), (1, 2)], [1, 2], (2, 3))
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

        np_array_1 = _np.array([5])
        np_array_2 = _np.array([[1, 2, 3]])
        array_1 = gs.array([5])
        array_2 = gs.array([[1, 2, 3]])

        np_result = _np.einsum('...,...i->...i', np_array_1, np_array_2)
        gs_result = gs.einsum('...,...i->...i', array_1, array_2)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array_1 = _np.array(5)
        np_array_2 = _np.array([[1, 2, 3]])
        array_1 = gs.array(5)
        array_2 = gs.array([[1, 2, 3]])

        np_result = _np.einsum('...,...i->...i', np_array_1, np_array_2)
        gs_result = gs.einsum('...,...i->...i', array_1, array_2)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array_1 = _np.array([5])
        np_array_2 = _np.array([1, 2, 3])
        array_1 = gs.array([5])
        array_2 = gs.array([1, 2, 3])

        np_result = _np.einsum('...,...i->...i', np_array_1, np_array_2)
        gs_result = gs.einsum('...,...i->...i', array_1, array_2)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array_1 = _np.array(5)
        np_array_2 = _np.array([1, 2, 3])
        array_1 = gs.array(5)
        array_2 = gs.array([1, 2, 3])

        np_result = _np.einsum('...,...i->...i', np_array_1, np_array_2)
        gs_result = gs.einsum('...,...i->...i', array_1, array_2)
        self.assertAllCloseToNp(gs_result, np_result)

    def test_einsum_dtypes(self):
        np_array_1 = _np.array([[1, 4]])
        np_array_2 = _np.array([[2., 3.]])
        array_1 = gs.array([[1, 4]])
        array_2 = gs.array([[2., 3.]])

        np_result = _np.einsum('...i,...i->...', np_array_1, np_array_2)
        gs_result = gs.einsum('...i,...i->...', array_1, array_2)

        self.assertAllCloseToNp(gs_result, np_result)

        np_array_1 = _np.array([[1., 4.], [-1., 5.]])
        np_array_2 = _np.array([[2, 3]])
        array_1 = gs.array([[1., 4.], [-1., 5.]])
        array_2 = gs.array([[2, 3]])

        np_result = _np.einsum('...i,...i->...', np_array_1, np_array_2)
        gs_result = gs.einsum('...i,...i->...', array_1, array_2)

        self.assertAllCloseToNp(gs_result, np_result)

    def test_assignment_with_matrices(self):
        np_array = _np.zeros((2, 3, 3))
        gs_array = gs.zeros((2, 3, 3))

        np_array[:, 0, 1] = 44.

        gs_array = gs.assignment(
            gs_array, 44., (0, 1), axis=0)

        self.assertAllCloseToNp(gs_array, np_array)

        n_samples = 3
        theta = _np.random.rand(5)
        phi = _np.random.rand(5)
        np_array = _np.zeros((n_samples, 5, 4))
        gs_array = gs.array(np_array)
        np_array[0, :, 0] = gs.cos(theta) * gs.cos(phi)
        np_array[0, :, 1] = - gs.sin(theta) * gs.sin(phi)
        gs_array = gs.assignment(
            gs_array, gs.cos(theta) * gs.cos(phi), (0, 0), axis=1)
        gs_array = gs.assignment(
            gs_array, - gs.sin(theta) * gs.sin(phi), (0, 1), axis=1)

        self.assertAllCloseToNp(gs_array, np_array)

    def test_assignment_with_booleans_single_index(self):
        np_array = _np.array([[2., 5.]])
        gs_array = gs.array([[2., 5.]])
        np_mask = _np.array([True])
        gs_mask = gs.array([True])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * _np.ones_like(np_array[~np_mask])
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs.ones_like(gs_array[~gs_mask]), ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([[2., 5.]])
        gs_array = gs.array([[2., 5.]])
        np_mask = _np.array([True])
        gs_mask = gs.array([True])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * np_array[~np_mask]
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs_array[~gs_mask], ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([[2., 5.]])
        gs_array = gs.array([[2., 5.]])
        np_mask = _np.array([False])
        gs_mask = gs.array([False])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * _np.ones_like(np_array[~np_mask])
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs.ones_like(gs_array[~gs_mask]), ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([[2., 5.]])
        gs_array = gs.array([[2., 5.]])
        np_mask = _np.array([False])
        gs_mask = gs.array([False])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * np_array[~np_mask]
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs_array[~gs_mask], ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

    def test_assignment_with_booleans_many_indices(self):
        np_array = _np.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        gs_array = gs.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])

        np_mask = _np.array([True, False, True])
        gs_mask = gs.array([True, False, True])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * _np.ones_like(np_array[~np_mask])
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs.ones_like(gs_array[~gs_mask]), ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        gs_array = gs.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])

        np_mask = _np.array([False, True, True])
        gs_mask = gs.array([False, True, True])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * _np.ones_like(np_array[~np_mask])
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs.ones_like(gs_array[~gs_mask]), ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        gs_array = gs.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        np_mask = _np.array([True, True, True])
        gs_mask = gs.array([True, True, True])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * _np.ones_like(np_array[~np_mask])
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs.ones_like(gs_array[~gs_mask]), ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        gs_array = gs.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        np_mask = _np.array([True, True, True])
        gs_mask = gs.array([True, True, True])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * np_array[~np_mask]
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs_array[~gs_mask], ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        gs_array = gs.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        np_mask = _np.array([False, False, False])
        gs_mask = gs.array([False, False, False])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * np_array[~np_mask]
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs_array[~gs_mask], ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

        np_array = _np.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        gs_array = gs.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        np_mask = _np.array([[False, False],
                            [False, True],
                            [True, True]])
        gs_mask = gs.array([[False, False],
                            [False, True],
                            [True, True]])

        np_array[np_mask] = _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] = 4 * np_array[~np_mask]
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment(
            gs_result, 4 * gs_array[~gs_mask], ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

    def test_assignment(self):
        gs_array_1 = gs.ones(3)
        self.assertRaises(
            ValueError, gs.assignment, gs_array_1, [.1, 2., 1.], [0, 1])

        np_array_1 = _np.ones(3)
        gs_array_1 = gs.ones_like(gs.array(np_array_1))

        np_array_1[2] = 1.5
        gs_result = gs.assignment(gs_array_1, 1.5, 2)
        self.assertAllCloseToNp(gs_result, np_array_1)

        np_array_1_list = _np.ones(3)
        gs_array_1_list = gs.ones_like(gs.array(np_array_1_list))

        indices = [1, 2]
        np_array_1_list[indices] = 1.5
        gs_result = gs.assignment(gs_array_1_list, 1.5, indices)
        self.assertAllCloseToNp(gs_result, np_array_1_list)

        np_array_2 = _np.zeros((3, 2))
        gs_array_2 = gs.zeros_like(gs.array(np_array_2))

        np_array_2[0, :] = 1
        gs_result = gs.assignment(gs_array_2, 1, 0, axis=1)
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

        gs_array_4_arr = gs.zeros_like(gs.array(np_array_4))

        gs_result = gs.assignment(gs_array_4_arr, 1, gs.array((0, 1)), axis=1)
        self.assertAllCloseToNp(gs_result, np_array_4)

        np_array_4_list = _np.zeros((3, 3, 2))
        gs_array_4_list = gs.zeros_like(gs.array(np_array_4_list))

        np_array_4_list[(0, 1), :, (1, 1)] = 1
        gs_result = gs.assignment(gs_array_4_list, 1, [(0, 1), (1, 1)], axis=1)
        self.assertAllCloseToNp(gs_result, np_array_4_list)

    def test_assignment_by_sum(self):
        gs_array_1 = gs.ones(3)
        self.assertRaises(
            ValueError, gs.assignment_by_sum, gs_array_1, [.1, 2., 1.], [0, 1])

        np_array_1 = _np.ones(3)
        gs_array_1 = gs.ones_like(gs.array(np_array_1))

        np_array_1[2] += 1.5
        gs_result = gs.assignment_by_sum(gs_array_1, 1.5, 2)
        self.assertAllCloseToNp(gs_result, np_array_1)

        gs_result_list = gs.assignment_by_sum(gs_array_1, [2., 1.5], [0, 2])
        np_array_1[0] += 2.
        self.assertAllCloseToNp(gs_result_list, np_array_1)

        np_array_1_list = _np.ones(3)
        gs_array_1_list = gs.ones_like(gs.array(np_array_1_list))

        indices = [1, 2]
        np_array_1_list[indices] += 1.5
        gs_result = gs.assignment_by_sum(gs_array_1_list, 1.5, indices)
        self.assertAllCloseToNp(gs_result, np_array_1_list)

        np_array_2 = _np.zeros((3, 2))
        gs_array_2 = gs.zeros_like(gs.array(np_array_2))

        np_array_2[0, :] += 1
        gs_result = gs.assignment_by_sum(gs_array_2, 1, 0, axis=1)
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

        np_array_4_list = _np.zeros((3, 3, 2))
        gs_array_4_list = gs.zeros_like(gs.array(np_array_4_list))

        np_array_4_list[(0, 1), :, (1, 1)] += 1
        gs_result = gs.assignment_by_sum(
            gs_array_4_list, 1, [(0, 1), (1, 1)], axis=1)
        self.assertAllCloseToNp(gs_result, np_array_4_list)

        n_samples = 3
        theta = _np.array([0.1, 0.2, 0.3, 0.4, 5.5])
        phi = _np.array([0.11, 0.22, 0.33, 0.44, -.55])
        np_array = _np.ones((n_samples, 5, 4))
        gs_array = gs.array(np_array)

        gs_array = gs.assignment_by_sum(
            gs_array, gs.cos(theta) * gs.cos(phi), (0, 0), axis=1)
        gs_array = gs.assignment_by_sum(
            gs_array, - gs.sin(theta) * gs.sin(phi), (0, 1), axis=1)

        np_array[0, :, 0] += _np.cos(theta) * _np.cos(phi)
        np_array[0, :, 1] -= _np.sin(theta) * _np.sin(phi)

        # TODO (ninamiolane): This test fails 15% of the time,
        # when gs and _np computations are in the reverse order.
        # We should investigate this.
        self.assertAllCloseToNp(gs_array, np_array)

        np_array = _np.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        gs_array = gs.array([
            [22., 55.],
            [33., 88.],
            [77., 99.]])
        np_mask = _np.array([[False, False],
                             [False, True],
                             [True, True]])
        gs_mask = gs.array([[False, False],
                            [False, True],
                            [True, True]])

        np_array[np_mask] += _np.zeros_like(np_array[np_mask])
        np_array[~np_mask] += 4 * np_array[~np_mask]
        np_result = np_array

        values_mask = gs.zeros_like(gs_array[gs_mask])
        gs_result = gs.assignment_by_sum(
            gs_array, values_mask, gs_mask)
        gs_result = gs.assignment_by_sum(
            gs_result, 4 * gs_array[~gs_mask], ~gs_mask)
        self.assertAllCloseToNp(gs_result, np_result)

    def test_any(self):
        base_list = [
            [[22., 55.],
             [33., 88.],
             [77., 99.]],
            [[34., 12.],
             [2., -3.],
             [67., 35.]]]
        np_array = _np.array(base_list)
        gs_array = gs.array(base_list)

        np_result = _np.any(np_array > 30.)
        gs_result = gs.any(gs_array > 30.)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.any(np_array > 30., axis=0)
        gs_result = gs.any(gs_array > 30., axis=0)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.any(np_array > 30., axis=-2)
        gs_result = gs.any(gs_array > 30., axis=-2)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.any(np_array > 30., axis=(-2, -1))
        gs_result = gs.any(gs_array > 30., axis=(-2, -1))
        self.assertAllCloseToNp(gs_result, np_result)

    def test_all(self):
        base_list = [
            [[22., 55.],
             [33., 88.],
             [77., 99.]],
            [[34., 12.],
             [2., -3.],
             [67., 35.]]]
        np_array = _np.array(base_list)
        gs_array = gs.array(base_list)

        np_result = _np.all(np_array > 30.)
        gs_result = gs.all(gs_array > 30.)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.all(np_array > 30., axis=0)
        gs_result = gs.all(gs_array > 30., axis=0)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.all(np_array > 30., axis=-2)
        gs_result = gs.all(gs_array > 30., axis=-2)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.all(np_array > 30., axis=(-2, -1))
        gs_result = gs.all(gs_array > 30., axis=(-2, -1))
        self.assertAllCloseToNp(gs_result, np_result)

    def test_trace(self):
        base_list = [
            [[22., 55.],
             [33., 88.]],
            [[34., 12.],
             [67., 35.]]]
        np_array = _np.array(base_list)
        gs_array = gs.array(base_list)

        np_result = _np.trace(np_array)
        gs_result = gs.trace(gs_array)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.trace(np_array, axis1=1, axis2=2)
        gs_result = gs.trace(gs_array, axis1=1, axis2=2)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.trace(np_array, axis1=-1, axis2=-2)
        gs_result = gs.trace(gs_array, axis1=-1, axis2=-2)
        self.assertAllCloseToNp(gs_result, np_result)

    def test_isclose(self):
        base_list = [
            [[22. + 1e-5, 22. + 1e-7],
             [22. + 1e-6, 88. + 1e-4]]]
        np_array = _np.array(base_list)
        gs_array = gs.array(base_list)

        np_result = _np.isclose(np_array, 22.)
        gs_result = gs.isclose(gs_array, 22.)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.isclose(np_array, 22.)
        gs_result = gs.isclose(gs_array, 22.)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.isclose(np_array, 22., rtol=1e-8, atol=1e-7)
        gs_result = gs.isclose(gs_array, 22., rtol=1e-8, atol=1e-7)
        self.assertAllCloseToNp(gs_result, np_result)

    @geomstats.tests.np_and_pytorch_only
    def test_where(self):
        # TODO (ninamiolane): Make tf behavior consistent with np
        # Currently, tf returns array, while np returns tuple
        base_list = [
            [[22., 55.],
             [33., 88.]],
            [[34., 12.],
             [67., 35.]]]
        np_array = _np.array(base_list)
        gs_array = gs.array(base_list)

        np_result = _np.where(np_array > 20., 0., np_array)
        gs_result = gs.where(gs_array > 20., 0., gs_array)
        self.assertAllCloseToNp(gs_result, np_result)

        np_result = _np.where(np_array > 20, np_array**2, 4.)
        gs_result = gs.where(gs_array > 20, gs_array**2, 4.)
        self.assertAllCloseToNp(gs_result, np_result)

        base_list = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
        np_array = _np.array(base_list)
        gs_array = gs.array(base_list)
        result = gs.where(gs_array == 0)
        expected = _np.where(np_array == 0)
        self.assertAllCloseToNp(*result, *expected)

        result = gs.where(gs_array == 0, - 1, gs_array)
        expected = _np.where(np_array == 0, - 1, np_array)
        self.assertAllCloseToNp(result, expected)

        expected = _np.where(np_array == 1, _np.ones(10), np_array)
        result = gs.where(gs_array == 1, gs.ones(10), gs_array)
        self.assertAllCloseToNp(result, expected)

    def test_convert_to_wider_dtype(self):
        gs_list = [gs.array([1, 2]), gs.array([2.2, 3.3], dtype=gs.float32)]
        gs_result = gs.convert_to_wider_dtype(gs_list)

        result = [a.dtype == gs.float32 for a in gs_result]

        self.assertTrue(gs.all(result))

        gs_list = [gs.array([1, 2]), gs.array([2.2, 3.3], dtype=gs.float64)]
        gs_result = gs.convert_to_wider_dtype(gs_list)

        result = [a.dtype == gs.float64 for a in gs_result]

        self.assertTrue(gs.all(result))

        gs_list = [
            gs.array([11.11, 222.2], dtype=gs.float64),
            gs.array([2.2, 3.3], dtype=gs.float32)]
        gs_result = gs.convert_to_wider_dtype(gs_list)

        result = [a.dtype == gs.float64 for a in gs_result]

        self.assertTrue(gs.all(result))

    def test_broadcast_arrays(self):

        array_1 = gs.array([[1, 2, 3]])
        array_2 = gs.array([[4], [5]])
        result = gs.broadcast_arrays(array_1, array_2)

        result_verdict = [gs.array([[1, 2, 3], [1, 2, 3]]),
                          gs.array([[4, 4, 4], [5, 5, 5]])]

        self.assertAllClose(result[0], result_verdict[0])
        self.assertAllClose(result[1], result_verdict[1])

        with self.assertRaises((ValueError, RuntimeError)):
            gs.broadcast_arrays(gs.array([1, 2]), gs.array([3, 4, 5]))

    def test_value_and_grad(self):
        n = 10
        vector = gs.ones(n)
        result_loss, result_grad = gs.autograd.value_and_grad(
            lambda v: gs.sum(v ** 2))(vector)
        expected_loss = n
        expected_grad = 2 * vector
        self.assertAllClose(result_loss, expected_loss)
        self.assertAllClose(result_grad, expected_grad)

    def test_value_and_grad_numpy_input(self):
        n = 10
        vector = _np.ones(n)
        result_loss, result_grad = gs.autograd.value_and_grad(
            lambda v: gs.sum(v ** 2))(vector)
        expected_loss = n
        expected_grad = 2 * vector
        self.assertAllClose(result_loss, expected_loss)
        self.assertAllClose(result_grad, expected_grad)

    def test_choice(self):
        x = gs.array([0.1, 0.2, 0.3, 0.4, 0.5])
        a = 4
        result = gs.random.choice(x, a)

        result_bool = True
        for i in result:
            if i in x:
                continue
            result_bool = False

        self.assertTrue(result_bool)
        self.assertEqual(len(result), a)

    def test_split(self):
        x = gs.array([0.1, 0.2, 0.3, 0.4])
        result = gs.split(x, 2)
        expected = _np.split(x, 2)
        for res, exp in zip(result, expected):
            self.assertAllClose(res, exp)

    def test_svd(self):
        gs_point = gs.reshape(gs.arange(12), (4, 3))
        gs_point = gs.cast(gs_point, gs.float64)
        np_point = _np.arange(12).reshape(4, 3)
        reconstruction = gs.array([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        u, s, v = _np.linalg.svd(np_point)
        u_r, s_r, v_r = gs.linalg.svd(gs_point)
        s_r_reconstructed = gs.einsum('kl,l->kl', reconstruction, s_r)
        gs_a_approx = gs.matmul(gs.matmul(
            u_r, s_r_reconstructed), v_r)
        s_reconstructed = _np.einsum('kl,l->kl', reconstruction, s)
        np_a_approx = _np.dot(u, _np.dot(s_reconstructed, v))
        self.assertAllClose(gs_a_approx, np_a_approx)

        full_matrices = False
        u, s, v = _np.linalg.svd(
            np_point, full_matrices=full_matrices)
        u_r, s_r, v_r = gs.linalg.svd(
            gs_point, full_matrices)
        reconstruction = gs.eye(3)
        s_r_reconstructed = gs.einsum('kl,l->kl', reconstruction, s_r)
        gs_a_approx = gs.matmul(gs.matmul(
            u_r, s_r_reconstructed), v_r)
        s_reconstructed = _np.einsum('kl,l->kl', reconstruction, s)
        np_a_approx = _np.dot(u, _np.dot(s_reconstructed, v))
        self.assertAllClose(gs_a_approx, np_a_approx)

        compute_uv = False
        s = _np.linalg.svd(np_point, compute_uv=compute_uv)
        s_r = gs.linalg.svd(gs_point, compute_uv=compute_uv)
        self.assertAllClose(s, s_r)

    def test_sylvester_solve(self):
        mat = gs.random.rand(4, 3)
        spd = gs.matmul(gs.transpose(mat), mat)

        mat = gs.random.rand(3, 3)
        skew = mat - gs.transpose(mat)
        solution = gs.linalg.solve_sylvester(spd, spd, skew)
        result = gs.matmul(spd, solution)
        result += gs.matmul(solution, spd)

        self.assertAllClose(result, skew)

    @geomstats.tests.np_and_pytorch_only
    def test_general_sylvester_solve(self):
        a = gs.array([[-3., -2., 0.], [-1., -1., 3.], [3., -5., -1.]])
        b = gs.array([[1.]])
        q = gs.array([[1.], [2.], [3.]])
        sol = gs.linalg.solve_sylvester(a, b, q)
        result = gs.matmul(a, sol) + gs.matmul(sol, b)
        self.assertAllClose(result, q)

    def test_sylvester_solve_vectorization(self):
        gs.random.seed(0)
        mat = gs.random.rand(2, 4, 3)
        spd = gs.matmul(gs.transpose(mat, (0, 2, 1)), mat)

        mat = gs.random.rand(2, 3, 3)
        skew = mat - gs.transpose(mat, (0, 2, 1))
        solution = gs.linalg.solve_sylvester(spd, spd, skew)
        result = gs.matmul(spd, solution)
        result += gs.matmul(solution, spd)

        self.assertAllClose(result, skew)

    def test_eigvalsh(self):
        mat = gs.array([[2., 1.], [1., -1.]])
        result = gs.linalg.eigvalsh(mat, UPLO='U')
        expected = _np.linalg.eigvalsh(mat)
        self.assertAllCloseToNp(result, expected)

    def test_cholesky(self):
        mat = SPDMatrices(3).random_point(2)
        result = gs.linalg.cholesky(mat)
        expected = _np.linalg.cholesky(mat)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_expm_backward(self):
        mat = gs.array([[0, 1, .5], [-1, 0, 0.2], [-.5, -.2, 0]])
        mat = gs.cast(mat, gs.float64)

        def loss(p):
            return gs.sum((gs.linalg.expm(p) - gs.eye(3)) ** 2)

        value_and_grad = gs.autograd.value_and_grad(loss)
        result = value_and_grad(mat)

        def loss_torch(p):
            return torch.sum((torch.matrix_exp(p) - torch.eye(3)) ** 2)

        torch_mat = torch.tensor(
            [[0, 1, .5], [-1, 0, 0.2], [-.5, -.2, 0]], dtype=torch.float64,
            requires_grad=True)
        value = loss_torch(torch_mat)
        value.backward()
        grad = torch_mat.grad

        self.assertAllClose(result[0], value.detach())
        self.assertAllClose(result[1], grad)

    def test_mat_from_diag_triu_tril(self):

        diag = gs.array([9., 9., 9.])
        triu = gs.array([1., 2., 3.])
        tril = -1 * triu
        mat = gs.mat_from_diag_triu_tril(
            diag, triu, tril)
        expected = gs.array([[9., 1., 2.], [-1., 9., 3., ], [-2., -3., 9.]])
        result = mat

        batch_diag = gs.eye(3)
        batch_triu = gs.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        batch_tril = -1 * batch_triu
        batch_mat = gs.mat_from_diag_triu_tril(
            batch_diag, batch_triu, batch_tril)

        batch_expected = gs.array([
            [[1., 1., 2.], [-1., 0., 3.], [-2., -3., 0.]],
            [[0., 4., 5.], [-4., 1., 6.], [-5., -6., 0.]],
            [[0., 7., 8.], [-7., 0., 9.], [-8., -9., 1.]]
        ])
        batch_result = batch_mat
        self.assertAllClose(expected, result)
        self.assertAllClose(batch_expected, batch_result)

    def test_prod(self):
        vec = gs.random.rand(10)
        result = gs.prod(vec)
        expected = gs.cumprod(vec)[-1]
        self.assertAllClose(result, expected)
