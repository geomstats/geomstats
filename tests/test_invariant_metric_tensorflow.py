"""
Unit tests for the invariant metrics on Lie groups
for tensorflow backend.
"""

import importlib
import os
import tensorflow as tf

import geomstats.backend as gs
import tests.helper as helper

from geomstats.invariant_metric import InvariantMetric
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


class TestInvariantMetricMethodsTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        n = 3
        group = SpecialOrthogonalGroup(n=n)

        # Diagonal left and right invariant metrics
        diag_mat_at_identity = gs.eye(group.dimension)

        left_diag_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='left')
        right_diag_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=diag_mat_at_identity,
                   left_or_right='right')

        # General left and right invariant metrics
        # TODO(nina): replace by general SPD matrix
        sym_mat_at_identity = gs.eye(group.dimension)

        left_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=sym_mat_at_identity,
                   left_or_right='left')

        right_metric = InvariantMetric(
                   group=group,
                   inner_product_mat_at_identity=sym_mat_at_identity,
                   left_or_right='right')

        metrics = {'left_diag': left_diag_metric,
                   'right_diag_metric': right_diag_metric,
                   'left': left_metric,
                   'right': right_metric}

        # General case for the point
        point_1 = tf.convert_to_tensor([-0.2, 0.9, 0.5])
        point_2 = tf.convert_to_tensor([0., 2., -0.1])
        # Edge case for the point, angle < epsilon,
        point_small = tf.convert_to_tensor([[-1e-7, 0., -7 * 1e-8]])

        self.group = group
        self.metrics = metrics

        self.left_diag_metric = left_diag_metric
        self.right_diag_metric = right_diag_metric
        self.left_metric = left_metric
        self.right_metric = right_metric
        self.point_1 = point_1
        self.point_2 = point_2
        self.point_small = point_small

    @classmethod
    def setUpClass(cls):
        #tf.enable_eager_execution()
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_inner_product_matrix(self):
        base_point = self.group.identity

        result = self.left_metric.inner_product_matrix(base_point=base_point)
        expected = self.left_metric.inner_product_mat_at_identity
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        result = self.right_metric.inner_product_matrix(base_point=base_point)
        expected = self.right_metric.inner_product_mat_at_identity
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_inner_product_matrix_and_its_inverse(self):
        inner_prod_mat = self.left_diag_metric.inner_product_mat_at_identity
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        result = gs.matmul(inv_inner_prod_mat, inner_prod_mat)
        expected = gs.eye(self.group.dimension)
        expected = gs.to_ndarray(expected, to_ndim=3, axis=0)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_left_exp_and_exp_from_identity_left_diag_metrics(self):
        left_exp_from_id = self.left_diag_metric.left_exp_from_identity(
                                                              self.point_1)
        exp_from_id = self.left_diag_metric.exp_from_identity(self.point_1)

        with self.test_session():
            self.assertAllClose(
                gs.eval(left_exp_from_id), gs.eval(exp_from_id))

    def test_left_log_and_log_from_identity_left_diag_metrics(self):
        left_log_from_id = self.left_diag_metric.left_log_from_identity(
                                                              self.point_1)
        log_from_id = self.left_diag_metric.log_from_identity(self.point_1)

        with self.test_session():
            self.assertAllClose(
                gs.eval(left_log_from_id), gs.eval(log_from_id))

    def test_left_exp_and_log_from_identity_left_diag_metrics(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # - exp then log
        # For left diagonal metric: point_1 and point_small
        result = helper.left_exp_then_log_from_identity(
                                        metric=self.left_diag_metric,
                                        tangent_vec=self.point_1)
        expected = self.point_1
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        result = helper.left_exp_then_log_from_identity(
                                        metric=self.left_diag_metric,
                                        tangent_vec=self.point_small)
        expected = self.point_small
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        # - log then exp

        # For left diagonal metric: point_1 and point_small
        result = helper.left_log_then_exp_from_identity(
                                        metric=self.left_diag_metric,
                                        point=self.point_1)
        expected = self.point_1
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        result = helper.left_log_then_exp_from_identity(
                                        metric=self.left_diag_metric,
                                        point=self.point_small)
        expected = self.point_small
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_left_exp_and_log_from_identity_left_metrics(self):
        """
        Test that the riemannian left exponential and the
        riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # - exp then log
        # For left metric: point_1 and point_small
        result = helper.left_exp_then_log_from_identity(
                                        metric=self.left_metric,
                                        tangent_vec=self.point_1)
        expected = self.point_1
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        result = helper.left_exp_then_log_from_identity(
                                        metric=self.left_metric,
                                        tangent_vec=self.point_small)
        expected = self.point_small
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        # - log then exp
        # For left metric: point_1 and point_small
        result = helper.left_log_then_exp_from_identity(
                                        metric=self.left_metric,
                                        point=self.point_1)
        expected = self.point_1
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        result = helper.left_log_then_exp_from_identity(
                                        metric=self.left_metric,
                                        point=self.point_small)
        expected = self.point_small
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    #def test_exp_and_log_from_identity_left_diag_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # - exp then log
    #    # For left diagonal metric, point and point_small
    #    result = helper.exp_then_log_from_identity(
    #                                    metric=self.left_diag_metric,
    #                                    tangent_vec=self.point_1)
    #    expected = self.point_1
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    result = helper.exp_then_log_from_identity(
    #                                    metric=self.left_diag_metric,
    #                                    tangent_vec=self.point_small)
    #    expected = self.point_small
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    # - log then exp
    #    # For left diagonal metric, point and point_small
    #    result = helper.log_then_exp_from_identity(
    #                                    metric=self.left_diag_metric,
    #                                    point=self.point_1)
    #    expected = self.point_1
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    result = helper.log_then_exp_from_identity(
    #                                    metric=self.left_diag_metric,
    #                                    point=self.point_small)
    #    expected = self.point_small
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #def test_exp_and_log_from_identity_left_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # - exp then log
    #    # For left metric, point and point_small
    #    result = helper.exp_then_log_from_identity(
    #                                    metric=self.left_metric,
    #                                    tangent_vec=self.point_1)
    #    expected = self.point_1
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    result = helper.exp_then_log_from_identity(
    #                                    metric=self.left_metric,
    #                                    tangent_vec=self.point_small)
    #    expected = self.point_small
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    # - log then exp
    #    # For left metric, point and point_small
    #    result = helper.log_then_exp_from_identity(
    #                                    metric=self.left_metric,
    #                                    point=self.point_1)
    #    expected = self.point_1
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    result = helper.log_then_exp_from_identity(
    #                                    metric=self.left_metric,
    #                                    point=self.point_small)
    #    expected = self.point_small
    #    expected = helper.to_vector(expected)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #def test_exp_and_log_from_identity_right_diag_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # TODO(nina): Fix this test.
    #    # - exp then log
    #    # For right diagonal metric, point and point_small
    #    result = helper.exp_then_log_from_identity(
    #                                    metric=self.right_diag_metric,
    #                                    tangent_vec=self.point_1)
    #    expected = self.point_1
    #    expected = helper.to_vector(expected)
    #    # with self.test_session():
    #    #     self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    result = helper.exp_then_log_from_identity(
    #                                    metric=self.right_diag_metric,
    #                                    tangent_vec=self.point_small)
    #    expected = self.point_small
    #    expected = helper.to_vector(expected)
    #    # with self.test_session():
    #    #     self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    # - log then exp
    #    # For right diagonal metric, point and point_small
    #    result = helper.log_then_exp_from_identity(
    #                                    metric=self.right_diag_metric,
    #                                    point=self.point_1)
    #    expected = self.point_1
    #    expected = helper.to_vector(expected)
    #    # with self.test_session():
    #    #     self.assertAllClose(gs.eval(result), gs.eval(expected))

    #    result = helper.log_then_exp_from_identity(
    #                                    metric=self.right_diag_metric,
    #                                    point=self.point_small)
    #    expected = self.point_small
    #    expected = helper.to_vector(expected)
    #    # with self.test_session():
    #    #     self.assertAllClose(gs.eval(result), gs.eval(expected))

    #def test_exp_and_log_from_identity_right_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # For right metric, point and point_small
    #    result = helper.exp_then_log_from_identity(self.right_metric,
    #                                               self.point_1)
    #    expected = self.point_1
    #    # self.assertTrue(gs.allclose(result, expected))

    #    result = helper.exp_then_log_from_identity(self.right_metric,
    #                                               self.point_small)
    #    expected = self.point_small
    #    # self.assertTrue(gs.allclose(result, expected))

    #    # - log then exp
    #    # For right metric, point and point_small
    #    result = helper.log_then_exp_from_identity(self.right_metric,
    #                                               self.point_1)
    #    expected = self.point_1
    #    # self.assertTrue(gs.allclose(result, expected))

    #    result = helper.log_then_exp_from_identity(self.right_metric,
    #                                               self.point_small)
    #    expected = self.point_small
    #    # self.assertTrue(gs.allclose(result, expected))

    #def test_exp_and_log_left_diag_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # General case for the reference point
    #    base_point = self.point_2

    #    # General point
    #    result = helper.log_then_exp(self.left_diag_metric,
    #                                 base_point, self.point_1)
    #    expected = self.group.regularize(self.point_1)
    #    # self.assertTrue(gs.allclose(result, expected))

    #    # Edge case, small angle
    #    result = helper.log_then_exp(self.left_diag_metric,
    #                                 base_point, self.point_small)
    #    expected = self.group.regularize(self.point_small)
    #    # self.assertTrue(gs.allclose(result, expected))

    #def test_exp_and_log_left_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # General case for the reference point
    #    base_point = self.point_2

    #    # For left metric: point and point_small
    #    result = helper.log_then_exp(self.left_metric,
    #                                 base_point, self.point_1)
    #    expected = self.point_1
    #    # self.assertTrue(gs.allclose(result, expected))

    #    result = helper.log_then_exp(self.left_metric,
    #                                 base_point, self.point_small)
    #    expected = self.point_small
    #    # self.assertTrue(gs.allclose(result, expected))

    #def test_exp_and_log_right_diag_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # General case for the reference point
    #    base_point = self.point_2
    #    # For right diagonal metric: point and point_small
    #    result = helper.log_then_exp(self.right_diag_metric,
    #                                 base_point, self.point_1)
    #    expected = self.group.regularize(self.point_1)
    #    # self.assertTrue(gs.allclose(result, expected))

    #    result = helper.log_then_exp(self.right_diag_metric,
    #                                 base_point, self.point_small)
    #    expected = self.group.regularize(self.point_small)
    #    # self.assertTrue(gs.allclose(result, expected))

    #def test_exp_and_log_right_metrics(self):
    #    """
    #    Test that the riemannian exponential and the
    #    riemannian logarithm are inverse.
    #    Expect their composition to give the identity function.
    #    """
    #    # General case for the reference point
    #    base_point = self.point_2
    #    # For right metric: point and point_small
    #    result = helper.log_then_exp(self.right_metric,
    #                                 base_point, self.point_1)
    #    expected = self.point_1
    #    # self.assertTrue(gs.allclose(result, expected))

    #    result = helper.log_then_exp(self.right_metric,
    #                                 base_point, self.point_small)
    #    expected = self.point_small
    #    # self.assertTrue(gs.allclose(result, expected))

    #def test_squared_dist_left_diag_metrics(self):
    #    sq_dist_1_2 = self.left_diag_metric.squared_dist(self.point_1,
    #                                                     self.point_2)
    #    sq_dist_2_1 = self.left_diag_metric.squared_dist(self.point_2,
    #                                                     self.point_1)
    #    # with self.test_session():
    #    #     self.assertAllClose(gs.eval(sq_dist_1_2), gs.eval(sq_dist_2_1))

    #def test_squared_dist_left_metrics(self):
    #    sq_dist_1_2 = self.left_metric.squared_dist(self.point_1,
    #                                                self.point_2)
    #    sq_dist_2_1 = self.left_metric.squared_dist(self.point_2,
    #                                                self.point_1)
    #    # with self.test_session():
    #    #     self.assertAllClose(gs.eval(sq_dist_1_2), gs.eval(sq_dist_2_1))

    #def test_squared_dist_and_squared_norm_left_diag_metrics(self):
    #    result = self.left_diag_metric.squared_dist(self.point_1,
    #                                                self.point_2)
    #    log = self.left_diag_metric.log(base_point=self.point_1,
    #                                    point=self.point_2)
    #    expected = self.left_diag_metric.squared_norm(
    #                                             vector=log,
    #                                             base_point=self.point_1)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #def test_squared_dist_and_squared_norm_left_metrics(self):
    #    result = self.left_metric.squared_dist(self.point_1,
    #                                           self.point_2)
    #    log = self.left_diag_metric.log(base_point=self.point_1,
    #                                    point=self.point_2)
    #    expected = self.left_metric.squared_norm(
    #                                         vector=log,
    #                                         base_point=self.point_1)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #def test_squared_dist_and_squared_norm_right_diag_metrics(self):
    #    result = self.right_diag_metric.squared_dist(self.point_1,
    #                                                 self.point_2)
    #    log = self.right_diag_metric.log(base_point=self.point_1,
    #                                     point=self.point_2)
    #    expected = self.right_diag_metric.squared_norm(
    #                                             vector=log,
    #                                             base_point=self.point_1)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))

    #def test_squared_dist_and_squared_norm_right_metrics(self):
    #    result = self.right_metric.squared_dist(self.point_1,
    #                                            self.point_2)
    #    log = self.right_diag_metric.log(base_point=self.point_1,
    #                                     point=self.point_2)
    #    expected = self.right_metric.squared_norm(
    #                                         vector=log,
    #                                         base_point=self.point_1)
    #    with self.test_session():
    #        self.assertAllClose(gs.eval(result), gs.eval(expected))


if __name__ == '__main__':
    tf.test.main()
