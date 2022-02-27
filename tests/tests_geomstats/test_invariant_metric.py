"""Unit tests for the invariant metrics on Lie groups."""

import itertools
import logging
import warnings
from asyncio import base_events

import pytest

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import TestCase
from tests.data_generation import LevelSetTestData, RiemannianMetricTestData, TestData
from tests.parametrizers import (
    LevelSetParametrizer,
    Parametrizer,
    RiemannianMetricParametrizer,
)


class TestInvariantMetric(TestCase, metaclass=Parametrizer):
    metric = connection = InvariantMetric

    class TestDataInvariantMetric(TestData):
        group = SpecialEuclidean(n=3, point_type="vector")
        matrix_se3 = SpecialEuclidean(n=3)
        matrix_so3 = SpecialOrthogonal(n=3)
        vector_so3 = SpecialOrthogonal(n=3, point_type="vector")
        point_1 = gs.array([-0.2, 0.9, 0.5, 5.0, 5.0, 5.0])
        point_2 = gs.array([0.0, 2.0, -0.1, 30.0, 400.0, 2.0])
        point_1_matrix = vector_so3.matrix_from_rotation_vector(point_1[..., :3])
        point_2_matrix = vector_so3.matrix_from_rotation_vector(point_2[..., :3])
        # Edge case for the point, angle < epsilon,
        point_small = gs.array([-1e-7, 0.0, -7 * 1e-8, 6.0, 5.0, 9.0])

        def inner_product_mat_at_identity_shape_data(self):
            group = SpecialEuclidean(n=3, point_type="vector")
            sym_mat_at_identity = gs.eye(group.dim)
            smoke_data = [
                dict(
                    group=group,
                    metric_mat_at_identity=sym_mat_at_identity,
                    left_or_right="left",
                )
            ]
            return self.generate_tests(smoke_data)

        def inner_product_matrix_shape_data(self):
            group = SpecialEuclidean(n=3, point_type="vector")
            sym_mat_at_identity = gs.eye(group.dim)
            smoke_data = [
                dict(
                    group=group,
                    metric_mat_at_identity=sym_mat_at_identity,
                    left_or_right="left",
                    base_point=None,
                ),
                dict(
                    group=group,
                    metric_mat_at_identity=sym_mat_at_identity,
                    left_or_right="left",
                    base_point=group.identity,
                ),
            ]
            return self.generate_tests(smoke_data)

        def inner_product_matrix_and_its_inverse_data(self):
            group = SpecialEuclidean(n=3, point_type="vector")
            smoke_data = [
                dict(group=group, metric_mat_at_identity=None, left_or_right="left")
            ]
            return self.generate_tests(smoke_data)

        def inner_product_at_identity_data(self):
            group = SpecialOrthogonal(n=3)
            algebra = group.lie_algebra
            tangent_vec_a = algebra.matrix_representation(gs.array([1.0, 0, 2.0]))
            tangent_vec_b = algebra.matrix_representation(gs.array([1.0, 0, 0.5]))
            batch_tangent_vec = algebra.matrix_representation(
                gs.array([[1.0, 0, 2.0], [0, 3.0, 5.0]])
            )
            smoke_data = [
                dict(
                    group=group,
                    metric_mat_at_identity=None,
                    left_or_right="left",
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=None,
                    expected=4.0,
                ),
                dict(
                    group=group,
                    metric_mat_at_identity=None,
                    left_or_right="left",
                    tangent_vec_a=batch_tangent_vec,
                    tangent_vec_b=tangent_vec_b,
                    base_point=None,
                    expected=gs.array([4.0, 5.0]),
                ),
                dict(
                    group=group,
                    metric_mat_at_identity=None,
                    left_or_right="left",
                    tangent_vec_a=group.compose(self.point_1_matrix, tangent_vec_a),
                    tangent_vec_b=group.compose(self.point_1_matrix, tangent_vec_b),
                    base_point=self.point_1_matrix,
                    expected=4.0,
                ),
                dict(
                    group=group,
                    metric_mat_at_identity=None,
                    left_or_right="left",
                    tangent_vec_a=group.compose(self.point_1_matrix, batch_tangent_vec),
                    tangent_vec_b=group.compose(self.point_1_matrix, tangent_vec_b),
                    base_point=self.point_1_matrix,
                    expected=gs.array([4.0, 5.0]),
                ),
                dict(
                    group=group,
                    metric_mat_at_identity=None,
                    left_or_right="right",
                    tangent_vec_a=group.compose(tangent_vec_a, self.point_1_matrix),
                    tangent_vec_b=group.compose(tangent_vec_b, self.point_1_matrix),
                    base_point=self.point_1_matrix,
                    expected=4.0,
                ),
                dict(
                    group=group,
                    metric_mat_at_identity=None,
                    left_or_right="right",
                    tangent_vec_a=group.compose(batch_tangent_vec, self.point_1_matrix),
                    tangent_vec_b=group.compose(tangent_vec_b, self.point_1_matrix),
                    base_point=self.point_1_matrix,
                    expected=gs.array([4.0, 5.0]),
                ),
            ]
            return self.generate_tests(smoke_data)

        def log_antipodals_data(self, group, rotation_mat1, rotation_mat2):
            smoke_data = [
                dict(
                    group=group,
                    rotation_mat1=gs.eye(3),
                    rotation_mat2=gs.array(
                        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
                    ),
                    expected=pytest.raises(ValueError),
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataInvariantMetric()

    def test_inner_product_mat_at_identity_shape(
        self, group, metric_mat_at_identity, left_or_right
    ):
        metric = self.metric(group, metric_mat_at_identity, left_or_right)
        dim = metric.group.dim
        result = self.left_metric.metric_mat_at_identity
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_shape(
        self, group, metric_mat_at_identity, left_or_right, base_point
    ):
        metric = self.metric(group, metric_mat_at_identity, left_or_right)
        base_point = None
        dim = self.left_metric.group.dim
        result = self.left_metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

        base_point = self.group.identity
        dim = self.left_metric.group.dim
        result = self.left_metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_and_its_inverse(
        self, group, metric_mat_at_identity, left_or_right
    ):
        metric = self.metric(group, metric_mat_at_identity, left_or_right)
        inner_prod_mat = metric.metric_mat_at_identity
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        result = gs.matmul(inv_inner_prod_mat, inner_prod_mat)
        expected = gs.eye(group.dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_inner_product(
        self,
        group,
        metric_mat_at_identity,
        left_or_right,
        tangent_vec_a,
        tangent_vec_b,
        base_point,
        expected,
    ):
        metric = self.metric(group, metric_mat_at_identity, left_or_right)
        result = metric.inner_product_at_identity(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(result, expected)

    def structure_constant_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group)
        x, y, z = metric.normal_basis(group.lie_algebra)
        smoke_data = []
        for x, y, z in itertools.permutations((x, y, z)):
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=x,
                    tangent_vec_b=y,
                    tangent_vec_c=z,
                    expected=2.0**0.5 / 2.0,
                )
            ]

        for x, y in itertools.permutations((x, y, z), 2):
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=x,
                    tangent_vec_b=x,
                    tangent_vec_c=y,
                    expected=0.0,
                )
            ]

        return self.generate_tests(smoke_data)

    def test_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        metric = InvariantMetric(group=group)
        result = metric.structure_constant(tangent_vec_a, tangent_vec_b, tangent_vec_c)
        self.assertAllClose(result, expected)

    def dual_adjoint_structure_constant_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group)
        x, y, z = metric.normal_basis(group.lie_algebra)
        smoke_data = []
        for x, y, z in itertools.permutations((x, y, z)):
            smoke_data += [
                dict(
                    group=group,
                    tangent_vec_a=x,
                    tangent_vec_b=y,
                    tangent_vec_c=z,
                )
            ]

        return self.generate_tests(smoke_data)

    def test_dual_adjoint_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        metric = InvariantMetric(group=group)
        result = metric.inner_product_at_identity(
            metric.dual_adjoint(tangent_vec_a, tangent_vec_b), tangent_vec_c
        )
        expected = metric.structure_constant(
            tangent_vec_a, tangent_vec_c, tangent_vec_b
        )
        self.assertAllClose(result, expected)

    def connection_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group)
        x, y, z = metric.normal_basis(group.lie_algebra.basis)
        smoke_data = [
            dict(
                group=group,
                tangent_vec_a=x,
                tangent_vec_b=y,
                expected=1.0 / 2**0.5 / 2.0 * z,
            )
        ]
        return self.generate_tests(smoke_data)

    def test_connection(self, group, tangent_vec_a, tangent_vec_b, expected):
        metric = InvariantMetric(group)
        self.assertAllClose(metric.connection(tangent_vec_a, tangent_vec_b), expected)

    def connection_translation_map_data(self, group):
        metric = InvariantMetric(group)
        smoke_data = [dict(group=group, point=group.random_point())]
        return self.generate_tests(smoke_data)

    def test_connection_translation_map(self, group, point):
        metric = InvariantMetric(group)
        translation_map = group.tangent_translation_map(point)
        x, y, z = metric.normal_basis(group.lie_algebra.basis)
        tan_a = translation_map(x)
        tan_b = translation_map(y)
        result = metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def sectional_curvature_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group)
        x, y, z = metric.normal_basis(group.lie_algebra.basis)
        smoke_data = [
            dict(group=group, tangent_vec_a=x, tangent_vec_b=y, expected=1.0 / 8),
            dict(group=group, tangent_vec_a=y, tangent_vec_b=y, expected=0.0),
            dict(
                group=group,
                tangent_vec_a=gs.stack([x, y]),
                tangent_vec_b=gs.stack([z] * 2),
                expected=gs.array([1.0 / 8, 1.0 / 8]),
            ),
        ]

    def test_sectional_curvature(self, group, tangent_vec_a, tangent_vec_b, expected):
        metric = InvariantMetric(group)
        result = metric.sectional_curvature(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(result, expected)

    def sectional_curvature_translation_point_data(self):
        return self.connection_translation_map_data()

    def test_sectional_curvature_translation_point(self, group, point):
        metric = InvariantMetric(group)
        translation_map = group.tangent_translation_map(point)
        x, y, z = metric.normal_basis(group.lie_algebra.basis)
        tan_a = translation_map(x)
        tan_b = translation_map(y)
        result = metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def curvature_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group)
        x, y, z = metric.normal_basis(group.lie_algebra.basis)
        smoke_data = [
            dict(
                group=group,
                tangent_vec_a=x,
                tangent_vec_b=y,
                tangent_vec_c=x,
                expected=1.0 / 8 * y,
            ),
            dict(
                group=group,
                tangent_vec_a=gs.stack([x, x]),
                tangent_vec_b=gs.stack([y] * 2),
                tangent_vec_c=gs.stack([x, x]),
                expected=gs.array([1.0 / 8 * y] * 2),
            ),
            dict(
                group=group,
                tangent_vec_a=y,
                tangent_vec_b=y,
                tangent_vec_c=z,
                expected=gs.zeros_like(z),
            ),
        ]
        return self.generate_tests(smoke_data)

    def test_curvature(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        metric = InvariantMetric(group)
        result = metric.sectional_curvature(tangent_vec_a, tangent_vec_b, tangent_vec_c)
        self.assertAllClose(result, expected)

    def curvature_translation_point_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group)
        x, y, z = metric.normal_basis(group.lie_algebra.basis)

        smoke_data = [
            dict(
                group=group,
                tangent_vec_a=x,
                tangent_vec_b=y,
                tangent_vec_c=x,
                point=group.random_point(),
            )
        ]
        return self.generate_tests(smoke_data)

    def test_curvature_translation_point(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, point, expected
    ):
        metric = InvariantMetric(group)
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        tan_c = translation_map(tangent_vec_c)
        result = metric.curvature(tan_a, tan_b, tan_c, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def test_curvature_derivative_at_identity(
        self,
        group,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        expected,
    ):
        metric = self.metric(group)
        result = metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d
        )

        self.assertAllClose(result, expected)

    def curvature_derivative_at_identity_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group)
        basis = metric.normal_basis(group.lie_algebra.basis)
        smoke_data = []
        for x in basis:
            for i, y in enumerate(basis):
                for z in basis[i:]:
                    for t in basis:
                        smoke_data.append(
                            dict(
                                group=group,
                                tangent_vec_a=x,
                                tangent_vec_b=y,
                                tangent_vec_c=z,
                                tangent_vec_d=t,
                                expected=gs.zeros_like(x),
                            )
                        )

        return self.generate_tests(smoke_data)

    def curvature_derivative_tangent_translation_map_data(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        x, y, z = metric.normal_basis(group.lie_algebra.basis)
        smoke_data = [
            dict(
                group=group,
                tangent_vec_a=x,
                tangent_vec_b=y,
                tangent_vec_c=z,
                tangent_vec_d=x,
                base_point=group.random_point(),
                expected=gs.zeros_like(x),
            )
        ]

    def test_curvature_derivative_tangent_translation_map(
        self,
        group,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point,
        expected,
    ):
        metric = InvariantMetric(group=group)
        translation_map = group.tangent_translation_map(base_point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        tan_c = translation_map(tangent_vec_c)
        tan_d = translation_map(tangent_vec_d)
        result = metric.curvature_derivative(tan_a, tan_b, tan_c, tan_d, base_point)
        self.assertAllClose(result, expected)

    def test_integrated_exp_at_id(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        basis = metric.normal_basis(group.lie_algebra.basis)

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = self.matrix_so3.identity
        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        expected = group.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-5)

        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk2")
        self.assertAllClose(expected, result, atol=1e-5)

    def test_integrated_exp_and_log_at_id(self):
        group = self.matrix_so3
        metric = InvariantMetric(group=group)
        basis = group.lie_algebra.basis

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = self.matrix_so3.identity

        exp = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        result = metric.log(exp, identity, n_steps=15, step="rk4", verbose=False)
        self.assertAllClose(tangent_vec, result, atol=1e-5)

    def test_integrated_se3_exp_at_id(self):
        group = self.matrix_se3
        lie_algebra = group.lie_algebra
        metric = InvariantMetric(group=group)
        canonical_metric = group.left_canonical_metric
        basis = metric.normal_basis(lie_algebra.basis)

        vector = gs.random.rand(len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = self.matrix_se3.identity
        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        expected = canonical_metric.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-5)

        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk2")
        self.assertAllClose(expected, result, atol=1e-5)

    def test_integrated_se3_exp(self):
        group = self.matrix_se3
        lie_algebra = group.lie_algebra
        metric = InvariantMetric(group=group)
        canonical_metric = group.left_canonical_metric
        basis = metric.normal_basis(lie_algebra.basis)
        point = group.random_point()

        vector = gs.random.rand(len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        tangent_vec = group.tangent_translation_map(point)(tangent_vec)
        result = metric.exp(tangent_vec, point, n_steps=100, step="rk4")
        expected = canonical_metric.exp(tangent_vec, point)
        self.assertAllClose(expected, result)

        result = metric.exp(tangent_vec, point, n_steps=100, step="rk2")
        self.assertAllClose(expected, result, atol=4e-5)

    @geomstats.tests.autograd_tf_and_torch_only
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

    def test_integrated_parallel_transport(self):
        group = self.matrix_se3
        metric = InvariantMetric(group=group)
        n = 3
        n_samples = 2

        point = group.identity
        tan_b = Matrices(n + 1, n + 1).random_point(n_samples)
        tan_b = group.to_tangent(tan_b)

        # use a vector orthonormal to tan_b
        tan_a = Matrices(n + 1, n + 1).random_point(n_samples)
        tan_a = group.to_tangent(tan_a)
        coef = metric.inner_product(tan_a, tan_b) / metric.squared_norm(tan_b)
        tan_a -= gs.einsum("...,...ij->...ij", coef, tan_b)
        tan_b = gs.einsum(
            "...ij,...->...ij", tan_b, 1.0 / metric.norm(tan_b, base_point=point)
        )
        tan_a = gs.einsum(
            "...ij,...->...ij", tan_a, 1.0 / metric.norm(tan_a, base_point=point)
        )

        expected = group.left_canonical_metric.parallel_transport(tan_a, point, tan_b)
        result, end_point_result = metric.parallel_transport(
            tan_a, point, tan_b, n_steps=20, step="rk4", return_endpoint=True
        )
        expected_end_point = metric.exp(tan_b, point, n_steps=20)

        self.assertAllClose(end_point_result, expected_end_point)
        self.assertAllClose(expected, result)

    def test_log_antipodals(self, group, rotation_mat1, rotation_mat2, expected):
        with expected:
            group.bi_invariant_metric.log(rotation_mat1, rotation_mat2)
