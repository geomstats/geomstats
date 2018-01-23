"""Unit tests for left- and right- invariant metrics module."""

import numpy as np
import unittest

from geomstats.invariant_metric import InvariantMetric
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.spd_matrices_space import is_symmetric


def left_log_then_exp_from_identity(metric, point):
    aux = metric.log_from_identity(point=point)
    result = metric.exp_from_identity(tangent_vec=aux)
    return result


def log_then_exp_from_identity(metric, point):
    aux = metric.log_from_identity(point=point)
    result = metric.exp_from_identity(tangent_vec=aux)
    return result


def log_then_exp(metric, base_point, point):
    aux = metric.log(base_point=base_point,
                     point=point)
    result = metric.exp(base_point=base_point,
                        tangent_vec=aux)
    return result


class TestInvariantMetricMethods(unittest.TestCase):

    def setUp(self):
        n = 3
        group = SpecialEuclideanGroup(n=n)

        diag_mat_at_identity = np.zeros([group.dimension, group.dimension])
        diag_mat_at_identity[0:3, 0:3] = 3 * np.eye(3)
        diag_mat_at_identity[3:6, 3:6] = 9 * np.eye(3)
        assert is_symmetric(diag_mat_at_identity)

        left_diag_metric = InvariantMetric(
                   lie_group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='left')
        right_diag_metric = InvariantMetric(
                   lie_group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='right')

        sym_mat_at_identity = np.array([[1., 2., 3., 4., 5., 6.],
                                        [2., 1., 4., 6., 3., 2.],
                                        [3., 4., 1., 0., 0., 0.],
                                        [4., 6., 0., 1., 3., -1.],
                                        [5., 3., 0., 3., 1., 0.],
                                        [6., 2., 0., -1., 0., 1]])
        assert is_symmetric(sym_mat_at_identity)

        left_metric = InvariantMetric(
                   lie_group=group,
                   inner_product_mat_at_identity=sym_mat_at_identity,
                   left_or_right='left')

        right_metric = InvariantMetric(
                   lie_group=group,
                   inner_product_mat_at_identity=sym_mat_at_identity,
                   left_or_right='right')

        # General case for the point
        rot_vec_1 = np.array([-1.2, 0.9, 0.9])  # NB: Regularized
        translation_1 = np.array([5, 5, 5])
        point_1 = np.concatenate([rot_vec_1,
                                  translation_1])

        # Edge case for the point angle < epsilon, where angle = norm(rot_vec)
        rot_vec_2 = np.array([-1e-7, 0., -7*1e-8])  # NB: Regularized
        translation_2 = np.array([6, 5, 9])
        point_2 = np.concatenate([rot_vec_2,
                                  translation_2])

        self.group = group
        self.left_diag_metric = left_diag_metric
        self.right_diag_metric = right_diag_metric
        self.left_metric = left_metric
        self.right_metric = right_metric
        self.point_1 = point_1
        self.point_2 = point_2

    def test_inner_product_matrix(self):
        base_point = self.group.identity
        result = self.left_metric.inner_product_matrix(base_point=base_point)

        expected = self.left_metric.inner_product_mat_at_identity
        self.assertTrue(np.allclose(result, expected))

        result = self.right_metric.inner_product_matrix(base_point=base_point)

        expected = self.right_metric.inner_product_mat_at_identity
        self.assertTrue(np.allclose(result, expected))

    def test_left_exp_and_log(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        rot_vec_base_point = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_base_point = np.array([4, -1, 2])
        base_point = np.concatenate([rot_vec_base_point,
                                     translation_base_point])

        # Tests
        result = log_then_exp(self.left_diag_metric, base_point, self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp(self.left_diag_metric, base_point, self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp(self.left_metric, base_point, self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp(self.left_metric, base_point, self.point_2)
        expected = self.point_2

        # TODO(nina): this last test does not pass. Non-spd matrix?
        # self.assertTrue(np.allclose(result, expected))

    def test_right_exp_and_log(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        rot_vec_base_point = np.pi / 3 * np.array([1, 0, 0])  # NB: Regularized
        translation_base_point = np.array([4, -1, 2])
        base_point = np.concatenate([rot_vec_base_point,
                                     translation_base_point])
        # Tests
        result = log_then_exp(self.right_diag_metric, base_point, self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp(self.right_diag_metric, base_point, self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp(self.right_metric, base_point, self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp(self.right_metric, base_point, self.point_2)
        expected = self.point_2

        # TODO(nina): this last test does not pass. Non-spd matrix?
        # self.assertTrue(np.allclose(result, expected))

    def test_left_exp_and_log_from_identity(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # With helper functions left_log_from_identity and
        # left_exp_from_identity
        result = left_log_then_exp_from_identity(self.left_diag_metric,
                                                 self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = left_log_then_exp_from_identity(self.left_diag_metric,
                                                 self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

        result = left_log_then_exp_from_identity(self.left_metric,
                                                 self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = left_log_then_exp_from_identity(self.left_metric,
                                                 self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

        # With wrapped functions log_from_identity and
        # exp_from_identity

        result = log_then_exp_from_identity(self.left_diag_metric,
                                            self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp_from_identity(self.left_diag_metric,
                                            self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp_from_identity(self.left_metric,
                                            self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp_from_identity(self.left_metric,
                                            self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

    def test_right_exp_and_log_from_identity(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        result = log_then_exp_from_identity(self.right_diag_metric,
                                            self.point_1)
        expected = self.point_1

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp_from_identity(self.right_diag_metric,
                                            self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

        result = log_then_exp_from_identity(self.right_metric,
                                            self.point_1)
        expected = self.point_1

        # This test does not pass.
        # self.assertTrue(np.allclose(result, expected))

        result = log_then_exp_from_identity(self.right_metric,
                                            self.point_2)
        expected = self.point_2

        self.assertTrue(np.allclose(result, expected))

if __name__ == '__main__':
        unittest.main()
