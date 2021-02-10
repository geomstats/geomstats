"""Unit tests for the invariant metrics on Lie groups."""

import logging
import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestInvariantMetric(geomstats.tests.TestCase):
    def setUp(self):
        logger = logging.getLogger()
        logger.disabled = True
        warnings.simplefilter('ignore', category=ImportWarning)

        gs.random.seed(1234)

        n = 3
        group = SpecialEuclidean(n=n, point_type='vector')
        matrix_se3 = SpecialEuclidean(n=n)
        matrix_so3 = SpecialOrthogonal(n=n)
        vector_so3 = SpecialOrthogonal(n=n, point_type='vector')

        # Diagonal left and right invariant metrics
        diag_mat_at_identity = gs.eye(group.dim)

        left_diag_metric = InvariantMetric(
            group=group,
            metric_mat_at_identity=None,
            left_or_right='left')
        right_diag_metric = InvariantMetric(
            group=group,
            metric_mat_at_identity=diag_mat_at_identity,
            left_or_right='right')

        sym_mat_at_identity = gs.eye(group.dim)

        left_metric = InvariantMetric(
            group=group,
            metric_mat_at_identity=sym_mat_at_identity,
            left_or_right='left')

        right_metric = InvariantMetric(
            group=group,
            metric_mat_at_identity=sym_mat_at_identity,
            left_or_right='right')

        matrix_left_metric = InvariantMetric(group=matrix_so3)

        matrix_right_metric = InvariantMetric(group=matrix_so3,
                                              left_or_right='right')

        # General case for the point
        point_1 = gs.array([-0.2, 0.9, 0.5, 5., 5., 5.])
        point_2 = gs.array([0., 2., -0.1, 30., 400., 2.])
        point_1_matrix = vector_so3.matrix_from_rotation_vector(
            point_1[..., :3])
        point_2_matrix = vector_so3.matrix_from_rotation_vector(
            point_2[..., :3])
        # Edge case for the point, angle < epsilon,
        point_small = gs.array([-1e-7, 0., -7 * 1e-8, 6., 5., 9.])

        self.group = group
        self.matrix_so3 = matrix_so3
        self.matrix_se3 = matrix_se3

        self.left_diag_metric = left_diag_metric
        self.right_diag_metric = right_diag_metric
        self.left_metric = left_metric
        self.right_metric = right_metric
        self.matrix_left_metric = matrix_left_metric
        self.matrix_right_metric = matrix_right_metric
        self.point_1 = point_1
        self.point_2 = point_2
        self.point_1_matrix = point_1_matrix
        self.point_2_matrix = point_2_matrix
        self.point_small = point_small

    def test_inner_product_mat_at_identity_shape(self):
        dim = self.left_metric.group.dim

        result = self.left_metric.metric_mat_at_identity
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_shape(self):
        base_point = None
        dim = self.left_metric.group.dim
        result = self.left_metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

        base_point = self.group.identity
        dim = self.left_metric.group.dim
        result = self.left_metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_and_inner_product_mat_at_identity(self):
        base_point = None
        result = self.left_metric.metric_matrix(base_point=base_point)
        expected = self.left_metric.metric_mat_at_identity
        self.assertAllClose(result, expected)

        base_point = self.group.identity
        result = self.right_metric.metric_matrix(base_point=base_point)
        expected = self.right_metric.metric_mat_at_identity
        self.assertAllClose(result, expected)

    def test_inner_product_matrix_and_its_inverse(self):
        inner_prod_mat = self.left_diag_metric.metric_mat_at_identity
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        result = gs.matmul(inv_inner_prod_mat, inner_prod_mat)
        expected = gs.eye(self.group.dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_inner_product_at_identity(self):
        lie_algebra = SkewSymmetricMatrices(3)
        tangent_vec_a = lie_algebra.matrix_representation(
            gs.array([1., 0, 2.]))
        tangent_vec_b = lie_algebra.matrix_representation(
            gs.array([1., 0, 0.5]))
        result = self.matrix_left_metric.inner_product_at_identity(
            tangent_vec_a, tangent_vec_b)
        expected = 4.
        self.assertAllClose(result, expected)

        tangent_vec_a = lie_algebra.matrix_representation(
            gs.array([[1., 0, 2.], [0, 3., 5.]]))
        result = self.matrix_left_metric.inner_product_at_identity(
            tangent_vec_a, tangent_vec_b)
        expected = gs.array([4., 5.])
        self.assertAllClose(result, expected)

    def test_inner_product_left(self):
        lie_algebra = SkewSymmetricMatrices(3)
        tangent_vec_a = lie_algebra.matrix_representation(
            gs.array([1., 0, 2.]))
        tangent_vec_a = self.matrix_so3.compose(
            self.point_1_matrix, tangent_vec_a)
        tangent_vec_b = lie_algebra.matrix_representation(
            gs.array([1., 0, 0.5]))
        tangent_vec_b = self.matrix_so3.compose(
            self.point_1_matrix, tangent_vec_b)
        result = self.matrix_left_metric.inner_product(
            tangent_vec_a, tangent_vec_b, self.point_1_matrix)
        expected = 4.
        self.assertAllClose(result, expected)

        tangent_vec_a = lie_algebra.matrix_representation(
            gs.array([[1., 0, 2.], [0, 3., 5.]]))
        tangent_vec_a = self.matrix_so3.compose(
            self.point_1_matrix, tangent_vec_a)
        result = self.matrix_left_metric.inner_product(
            tangent_vec_a, tangent_vec_b, self.point_1_matrix)
        expected = gs.array([4., 5.])
        self.assertAllClose(result, expected)

    def test_inner_product_right(self):
        lie_algebra = SkewSymmetricMatrices(3)
        tangent_vec_a = lie_algebra.matrix_representation(
            gs.array([1., 0, 2.]))
        tangent_vec_a = self.matrix_so3.compose(
            tangent_vec_a, self.point_1_matrix)
        tangent_vec_b = lie_algebra.matrix_representation(
            gs.array([1., 0, 0.5]))
        tangent_vec_b = self.matrix_so3.compose(
            tangent_vec_b, self.point_1_matrix)
        result = self.matrix_right_metric.inner_product(
            tangent_vec_a, tangent_vec_b, self.point_1_matrix)
        expected = 4.
        self.assertAllClose(result, expected)

        tangent_vec_a = lie_algebra.matrix_representation(
            gs.array([[1., 0, 2.], [0, 3., 5.]]))
        tangent_vec_a = self.matrix_so3.compose(
            tangent_vec_a, self.point_1_matrix)
        result = self.matrix_right_metric.inner_product(
            tangent_vec_a, tangent_vec_b, self.point_1_matrix)
        expected = gs.array([4., 5.])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_left_exp_and_exp_from_identity_left_diag_metrics(self):
        left_exp_from_id = self.left_diag_metric.left_exp_from_identity(
            self.point_1)
        exp_from_id = self.left_diag_metric.exp_from_identity(self.point_1)

        self.assertAllClose(left_exp_from_id, exp_from_id)

    @geomstats.tests.np_and_tf_only
    def test_left_log_and_log_from_identity_left_diag_metrics(self):
        left_log_from_id = self.left_diag_metric.left_log_from_identity(
            self.point_1)
        log_from_id = self.left_diag_metric.log_from_identity(self.point_1)

        self.assertAllClose(left_log_from_id, log_from_id)

    @geomstats.tests.np_and_tf_only
    def test_left_exp_and_log_from_identity_left_diag_metrics(self):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # - exp then log
        # For left diagonal metric: point_1 and point_small
        result = helper.left_exp_then_log_from_identity(
            metric=self.left_diag_metric,
            tangent_vec=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.left_exp_then_log_from_identity(
            metric=self.left_diag_metric,
            tangent_vec=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

        # - log then exp

        # For left diagonal metric: point_1 and point_small
        result = helper.left_log_then_exp_from_identity(
            metric=self.left_diag_metric,
            point=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.left_log_then_exp_from_identity(
            metric=self.left_diag_metric,
            point=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_left_exp_and_log_from_identity_left_metrics(self):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # - exp then log
        # For left metric: point_1 and point_small
        result = helper.left_exp_then_log_from_identity(
            metric=self.left_metric,
            tangent_vec=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.exp_then_log(
            metric=self.left_metric,
            tangent_vec=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.left_exp_then_log_from_identity(
            metric=self.left_metric,
            tangent_vec=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

        # - log then exp
        # For left metric: point_1 and point_small
        result = helper.left_log_then_exp_from_identity(
            metric=self.left_metric,
            point=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.left_log_then_exp_from_identity(
            metric=self.left_metric,
            point=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_from_identity_left_diag_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # - exp then log
        # For left diagonal metric, point and point_small
        result = helper.exp_then_log_from_identity(
            metric=self.left_diag_metric,
            tangent_vec=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.exp_then_log_from_identity(
            metric=self.left_diag_metric,
            tangent_vec=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

        # - log then exp
        # For left diagonal metric, point and point_small
        result = helper.log_then_exp_from_identity(
            metric=self.left_diag_metric,
            point=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.log_then_exp_from_identity(
            metric=self.left_diag_metric,
            point=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_from_identity_left_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # - exp then log
        # For left metric, point and point_small
        result = helper.exp_then_log_from_identity(
            metric=self.left_metric,
            tangent_vec=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.exp_then_log_from_identity(
            metric=self.left_metric,
            tangent_vec=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

        # - log then exp
        # For left metric, point and point_small
        result = helper.log_then_exp_from_identity(
            metric=self.left_metric,
            point=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.log_then_exp_from_identity(
            metric=self.left_metric,
            point=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_from_identity_right_diag_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # - exp then log
        # For right diagonal metric, point and point_small
        result = helper.exp_then_log_from_identity(
            metric=self.right_diag_metric,
            tangent_vec=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.exp_then_log_from_identity(
            metric=self.right_diag_metric,
            tangent_vec=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

        # - log then exp
        # For right diagonal metric, point and point_small
        result = helper.log_then_exp_from_identity(
            metric=self.right_diag_metric,
            point=self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.log_then_exp_from_identity(
            metric=self.right_diag_metric,
            point=self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_from_identity_right_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # For right metric, point and point_small
        result = helper.exp_then_log_from_identity(
            self.right_metric, self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.exp_then_log_from_identity(
            self.right_metric, self.point_small)
        expected = self.point_small
        # self.assertAllClose(result, expected)

        # - log then exp
        # For right metric, point and point_small
        result = helper.log_then_exp_from_identity(
            self.right_metric, self.point_1)
        expected = self.point_1
        self.assertAllClose(result, expected)

        result = helper.log_then_exp_from_identity(
            self.right_metric, self.point_small)
        expected = self.point_small
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_left_diag_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        base_point = self.point_2
        # General point
        result = helper.log_then_exp(
            self.left_diag_metric, self.point_1, base_point)
        expected = self.group.regularize(self.point_1)
        result = self.group.regularize(result)
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

        # Edge case, small angle
        result = helper.log_then_exp(
            self.left_diag_metric, self.point_small, base_point)
        expected = self.group.regularize(self.point_small)
        result = self.group.regularize(result)
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_left_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        base_point = self.point_2

        # For left metric: point and point_small
        result = helper.log_then_exp(
            self.left_metric, self.point_1, base_point)
        expected = self.point_1
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

        result = helper.log_then_exp(
            self.left_metric, self.point_small, base_point)
        expected = self.point_small
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_right_diag_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        base_point = self.point_2
        # For right diagonal metric: point and point_small
        result = helper.log_then_exp(
            self.right_diag_metric, self.point_1, base_point)
        expected = self.group.regularize(self.point_1)
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

        result = helper.log_then_exp(
            self.right_diag_metric, self.point_small, base_point)
        expected = self.group.regularize(self.point_small)
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_right_metrics(self):
        """
        Test that the Riemannian exponential and the
        Riemannian logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # General case for the reference point
        base_point = self.point_2
        # For right metric: point and point_small
        result = helper.log_then_exp(
            self.right_metric, self.point_1, base_point)
        expected = self.point_1
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

        result = helper.log_then_exp(
            self.right_metric, self.point_small, base_point)
        expected = self.point_small
        self.assertAllClose(result, expected, atol=1e-4, rtol=1e-4)

    @geomstats.tests.np_and_tf_only
    def test_squared_dist_left_diag_metrics(self):
        sq_dist_1_2 = self.left_diag_metric.squared_dist(self.point_1,
                                                         self.point_2)
        sq_dist_2_1 = self.left_diag_metric.squared_dist(self.point_2,
                                                         self.point_1)
        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

    @geomstats.tests.np_and_tf_only
    def test_squared_dist_left_metrics(self):
        sq_dist_1_2 = self.left_metric.squared_dist(self.point_1,
                                                    self.point_2)
        sq_dist_2_1 = self.left_metric.squared_dist(self.point_2,
                                                    self.point_1)
        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

    @geomstats.tests.np_and_tf_only
    def test_squared_dist_and_squared_norm_left_diag_metrics(self):
        result = self.left_diag_metric.squared_dist(self.point_1,
                                                    self.point_2)
        log = self.left_diag_metric.log(base_point=self.point_1,
                                        point=self.point_2)
        expected = self.left_diag_metric.squared_norm(
            vector=log,
            base_point=self.point_1)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_squared_dist_and_squared_norm_left_metrics(self):
        result = self.left_metric.squared_dist(self.point_1, self.point_2)
        log = self.left_metric.log(base_point=self.point_1, point=self.point_2)
        expected = self.left_metric.squared_norm(
            vector=log, base_point=self.point_1)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_squared_dist_and_squared_norm_right_diag_metrics(self):
        result = self.right_diag_metric.squared_dist(self.point_1,
                                                     self.point_2)
        log = self.right_diag_metric.log(base_point=self.point_1,
                                         point=self.point_2)
        expected = self.right_diag_metric.squared_norm(
            vector=log, base_point=self.point_1)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_squared_dist_and_squared_norm_right_metrics(self):
        result = self.right_metric.squared_dist(self.point_1, self.point_2)
        log = self.right_metric.log(
            base_point=self.point_1, point=self.point_2)
        expected = self.right_metric.squared_norm(
            vector=log, base_point=self.point_1)
        self.assertAllClose(result, expected)

    def test_structure_constant(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        basis = metric.orthonormal_basis(group.lie_algebra.basis)
        x, y, z = basis
        result = metric.structure_constant(x, y, z)
        expected = 2. ** .5 / 2.
        self.assertAllClose(result, expected)

        result = -metric.structure_constant(y, x, z)
        self.assertAllClose(result, expected)

        result = metric.structure_constant(y, z, x)
        self.assertAllClose(result, expected)

        result = -metric.structure_constant(z, y, x)
        self.assertAllClose(result, expected)

        result = metric.structure_constant(z, x, y)
        self.assertAllClose(result, expected)

        result = -metric.structure_constant(x, z, y)
        self.assertAllClose(result, expected)

        result = metric.structure_constant(x, x, z)
        expected = 0.
        self.assertAllClose(result, expected)

    def test_dual_adjoint(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        basis = metric.orthonormal_basis(group.lie_algebra.basis)
        for x in basis:
            for y in basis:
                for z in basis:
                    result = metric.inner_product_at_identity(
                        metric.dual_adjoint(x, y), z)
                    expected = metric.structure_constant(x, z, y)
                    self.assertAllClose(result, expected)

    def test_connection(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        x, y, z = metric.orthonormal_basis(group.lie_algebra.basis)
        result = metric.connection(x, y)
        expected = 1. / 2 ** .5 / 2. * z
        self.assertAllClose(result, expected)

        point = group.random_uniform()
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(x)
        tan_b = translation_map(y)
        result = metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def test_sectional_curvature(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        x, y, z = metric.orthonormal_basis(group.lie_algebra.basis)

        result = metric.sectional_curvature(x, y)
        expected = 1. / 8
        self.assertAllClose(result, expected)

        point = group.random_uniform()
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(x)
        tan_b = translation_map(y)
        result = metric.sectional_curvature(tan_a, tan_b, point)
        self.assertAllClose(result, expected)

        tan_a = gs.stack([x, y])
        tan_b = gs.stack([z] * 2)
        result = metric.sectional_curvature(tan_a, tan_b)
        self.assertAllClose(result, gs.array([expected] * 2))

        result = metric.sectional_curvature(y, y)
        expected = 0.
        self.assertAllClose(result, expected)

    def test_curvature(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        x, y, z = metric.orthonormal_basis(group.lie_algebra.basis)

        result = metric.curvature_at_identity(x, y, x)
        expected = 1. / 8 * y
        self.assertAllClose(result, expected)

        tan_a = gs.stack([x, x])
        tan_b = gs.stack([y] * 2)
        result = metric.curvature(tan_a, tan_b, tan_a)
        self.assertAllClose(result, gs.array([expected] * 2))

        point = group.random_uniform()
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(x)
        tan_b = translation_map(y)
        result = metric.curvature(tan_a, tan_b, tan_a, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

        result = metric.curvature(y, y, z)
        expected = gs.zeros_like(z)
        self.assertAllClose(result, expected)

    def test_curvature_derivative_at_identity(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        basis = metric.orthonormal_basis(group.lie_algebra.basis)

        result = True
        for x in basis:
            for i, y in enumerate(basis):
                for z in basis[i:]:
                    for t in basis:
                        nabla_r = metric.curvature_derivative_at_identity(
                            x, y, z, t)
                        if not gs.all(gs.isclose(nabla_r, 0., atol=1e-5)):
                            print(nabla_r)
                            result = False
        self.assertTrue(result)

    def test_curvature_derivative(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        x, y, z = metric.orthonormal_basis(group.lie_algebra.basis)
        result = metric.curvature_derivative(
            x, y, z, x)
        expected = gs.zeros_like(x)
        self.assertAllClose(result, expected)

        point = group.random_uniform()
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(x)
        tan_b = translation_map(y)
        tan_c = translation_map(z)
        result = metric.curvature_derivative(
            tan_a, tan_b, tan_c, tan_a, point)
        expected = gs.zeros_like(x)
        self.assertAllClose(result, expected)

    def test_integrated_exp_at_id(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        basis = metric.orthonormal_basis(group.lie_algebra.basis)

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum('...j,jkl->...kl', vector, basis)
        identity = self.matrix_so3.identity
        result = metric.exp(
            tangent_vec, identity, n_steps=100, step='rk4')
        expected = group.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-5)

        result = metric.exp(
            tangent_vec, identity, n_steps=100, step='rk2')
        self.assertAllClose(expected, result, atol=1e-5)

    def test_integrated_exp_and_log_at_id(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        basis = group.lie_algebra.basis

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum('...j,jkl->...kl', vector, basis)
        identity = self.matrix_so3.identity
        exp = metric.exp(
            tangent_vec, identity, n_steps=100, step='rk4')
        result = metric.log(
            exp, identity,
            n_steps=15, step='rk4', verbose=False)
        self.assertAllClose(tangent_vec, result, atol=1e-5)

    def test_integrated_se3_exp_at_id(self):
        group = self.matrix_se3
        lie_algebra = group.lie_algebra
        metric = InvariantMetric(group=group)
        canonical_metric = group.left_canonical_metric
        basis = metric.orthonormal_basis(lie_algebra.basis)

        vector = gs.random.rand(len(basis))
        tangent_vec = gs.einsum('...j,jkl->...kl', vector, basis)
        identity = self.matrix_se3.identity
        result = metric.exp(
            tangent_vec, identity, n_steps=100, step='rk4')
        expected = canonical_metric.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-5)

        result = metric.exp(
            tangent_vec, identity, n_steps=100, step='rk2')
        self.assertAllClose(expected, result, atol=1e-5)

    def test_integrated_se3_exp(self):
        group = self.matrix_se3
        lie_algebra = group.lie_algebra
        metric = InvariantMetric(group=group)
        canonical_metric = group.left_canonical_metric
        basis = metric.orthonormal_basis(lie_algebra.basis)
        point = group.random_uniform()

        vector = gs.random.rand(len(basis))
        tangent_vec = gs.einsum('...j,jkl->...kl', vector, basis)
        tangent_vec = group.tangent_translation_map(point)(tangent_vec)
        result = metric.exp(
            tangent_vec, point, n_steps=100, step='rk4')
        expected = canonical_metric.exp(tangent_vec, point)
        self.assertAllClose(expected, result)

        result = metric.exp(
            tangent_vec, point, n_steps=100, step='rk2')
        self.assertAllClose(expected, result, atol=4e-5)

    def test_dist_pairwise_parallel(self):
        gs.random.seed(0)
        n_samples = 2
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        points = group.random_uniform(n_samples)
        result = metric.dist_pairwise(points, n_jobs=2)
        is_sym = Matrices.is_symmetric(result)
        belongs = Matrices(n_samples, n_samples).belongs(result)
        self.assertTrue(is_sym)
        self.assertTrue(belongs)
