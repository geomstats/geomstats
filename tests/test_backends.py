"""Unit tests for backends.

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

        #print('numpy')
        add_base_point = _np.array([[-0.02064695, 0.26152784]])
        norm_add = _np.array([[0.2623416]])

        mask_0 = _np.isclose(_np.squeeze(norm_add, axis=-1), 0.)
        mask_non0 = ~mask_0
        add_base_point[mask_0] = _np.zeros_like(add_base_point[mask_0])
        #print('add_base_point', add_base_point)
        add_base_point[mask_non0] = add_base_point[mask_non0] / norm_add[mask_non0]
        #print('add_base_point', add_base_point)
        np_result = add_base_point

        #print('backend')
        add_base_point = gs.array([[-0.02064695, 0.26152784]])
        norm_add = gs.array([[0.2623416]])

        mask_0 = gs.isclose(gs.squeeze(norm_add, axis=-1), 0.)
        mask_non0 = ~mask_0
        add_base_point = gs.assignment(
            add_base_point,
            gs.zeros_like(add_base_point[mask_0]),
            mask_0)
        #print('add_base_point', add_base_point)
        add_base_point = gs.assignment(
            add_base_point,
            add_base_point[mask_non0] / norm_add[mask_non0],
            mask_non0)
        #print('add_base_point', add_base_point)
        gs_result = add_base_point
        #print('gs_result', gs_result)
        #print('np_result', np_result)
        #print(gs_result == np_result)
        #print(_np.all(gs_result == np_result))
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

    def test_assignment(self):
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
        print(gs_result)
        print(np_array_1)
        self.assertAllCloseToNp(gs_result, np_array_1_list)

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

        np_array_4_list = _np.zeros((3, 3, 2))
        gs_array_4_list = gs.zeros_like(gs.array(np_array_4_list))

        np_array_4_list[(0, 1), :, (1, 1)] = 1
        gs_result = gs.assignment(gs_array_4_list, 1, [(0, 1), (1, 1)], axis=1)
        self.assertAllCloseToNp(gs_result, np_array_4_list)

    def test_assignment_by_sum(self):
        np_array_1 = _np.ones(3)
        gs_array_1 = gs.ones_like(gs.array(np_array_1))

        np_array_1[2] += 1.5
        gs_result = gs.assignment_by_sum(gs_array_1, 1.5, 2)
        self.assertAllCloseToNp(gs_result, np_array_1)

        np_array_1_list = _np.ones(3)
        gs_array_1_list = gs.ones_like(gs.array(np_array_1_list))

        indices = [1, 2]
        np_array_1_list[indices] += 1.5
        gs_result = gs.assignment_by_sum(gs_array_1_list, 1.5, indices)
        self.assertAllCloseToNp(gs_result, np_array_1_list)

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

        np_array_4_list = _np.zeros((3, 3, 2))
        gs_array_4_list = gs.zeros_like(gs.array(np_array_4_list))

        np_array_4_list[(0, 1), :, (1, 1)] += 1
        gs_result = gs.assignment_by_sum(
            gs_array_4_list, 1, [(0, 1), (1, 1)], axis=1)
        self.assertAllCloseToNp(gs_result, np_array_4_list)
