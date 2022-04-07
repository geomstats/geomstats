"""Unit tests for the invariant metrics on Lie groups."""

import itertools

import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import Parametrizer, np_backend
from tests.data_generation import _RiemannianMetricTestData
from tests.geometry_test_cases import RiemannianMetricTestCase


class TestInvariantMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = InvariantMetric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = np_backend()
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_ladder_parallel_transport = np_backend()
    skip_test_log_is_tangent = np_backend()
    skip_test_log_shape = np_backend()
    skip_test_geodesic_bvp_belongs = np_backend()
    skip_test_exp_after_log = np_backend()
    skip_test_geodesic_bvp_belongs = True
    skip_test_log_after_exp = True
    skip_test_dist_point_to_itself_is_zero = True

    class InvariantMetricTestData(_RiemannianMetricTestData):
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

        diag_mat_at_identity = gs.eye(group.dim)
        metric_args_list = [
            (group, None, "left"),
            (group, None, "right"),
            (group, gs.eye(group.dim), "left"),
            (group, gs.eye(group.dim), "right"),
            (matrix_so3, None, "right"),
            (matrix_so3, None, "left"),
        ]
        shape_list = [metric_args[0].shape for metric_args in metric_args_list]
        space_list = [metric_args[0] for metric_args in metric_args_list]
        n_points_list = [1, 2] * 3
        n_tangent_vecs_list = [1, 2] * 3
        n_points_a_list = [1, 2] * 3
        n_points_b_list = [1]
        alpha_list = [1] * 6
        n_rungs_list = [1] * 6
        scheme_list = ["pole"] * 6

        def inner_product_mat_at_identity_shape_test_data(self):
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

        def inner_product_matrix_shape_test_data(self):
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

        def inner_product_matrix_and_its_inverse_test_data(self):
            group = SpecialEuclidean(n=3, point_type="vector")
            smoke_data = [
                dict(group=group, metric_mat_at_identity=None, left_or_right="left")
            ]
            return self.generate_tests(smoke_data)

        def inner_product_test_data(self):
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

        def log_antipodals_test_data(self):
            group = self.matrix_so3
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

        def structure_constant_test_data(self):
            group = self.matrix_so3
            metric = InvariantMetric(group)
            x, y, z = metric.normal_basis(group.lie_algebra.basis)
            smoke_data = []
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=x,
                    tangent_vec_b=y,
                    tangent_vec_c=z,
                    expected=2.0**0.5 / 2.0,
                )
            ]
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=y,
                    tangent_vec_b=x,
                    tangent_vec_c=z,
                    expected=-(2.0**0.5 / 2.0),
                )
            ]
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=y,
                    tangent_vec_b=z,
                    tangent_vec_c=x,
                    expected=2.0**0.5 / 2.0,
                )
            ]
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=z,
                    tangent_vec_b=y,
                    tangent_vec_c=x,
                    expected=-(2.0**0.5 / 2.0),
                )
            ]
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=z,
                    tangent_vec_b=x,
                    tangent_vec_c=y,
                    expected=2.0**0.5 / 2.0,
                )
            ]
            smoke_data += [
                dict(
                    group=self.matrix_so3,
                    tangent_vec_a=x,
                    tangent_vec_b=z,
                    tangent_vec_c=y,
                    expected=-(2.0**0.5 / 2.0),
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

        def dual_adjoint_structure_constant_test_data(self):
            group = self.matrix_so3
            metric = InvariantMetric(group)
            x, y, z = metric.normal_basis(group.lie_algebra.basis)
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

        def connection_test_data(self):
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

        def connection_translation_map_test_data(self):
            group = self.matrix_so3
            metric = InvariantMetric(group)
            x, y, z = metric.normal_basis(group.lie_algebra.basis)
            smoke_data = [
                dict(
                    group=group,
                    tangent_vec_a=x,
                    tangent_vec_b=y,
                    point=group.random_point(),
                    expected=1.0 / 2**0.5 / 2.0 * z,
                )
            ]
            return self.generate_tests(smoke_data)

        def sectional_curvature_test_data(self):
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
            return self.generate_tests(smoke_data)

        def sectional_curvature_translation_point_test_data(self):
            return self.connection_translation_map_test_data()

        def curvature_test_data(self):
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

        def curvature_translation_point_test_data(self):
            group = self.matrix_so3
            metric = InvariantMetric(group)
            x, y, _ = metric.normal_basis(group.lie_algebra.basis)

            smoke_data = [
                dict(
                    group=group,
                    tangent_vec_a=x,
                    tangent_vec_b=y,
                    tangent_vec_c=x,
                    point=group.random_point(),
                    expected=1.0 / 8 * y,
                )
            ]
            return self.generate_tests(smoke_data)

        def curvature_derivative_at_identity_test_data(self):
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

        def curvature_derivative_tangent_translation_map_test_data(self):
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
            return self.generate_tests(smoke_data)

        def integrated_exp_at_id_test_data(self):

            smoke_data = [dict(group=self.matrix_so3)]
            return self.generate_tests(smoke_data)

        def integrated_se3_exp_at_id_test_data(self):
            smoke_data = [dict(group=self.matrix_se3)]
            return self.generate_tests(smoke_data)

        def integrated_exp_and_log_at_id_test_data(self):
            smoke_data = [dict(group=self.matrix_so3)]
            return self.generate_tests(smoke_data)

        def integrated_parallel_transport_test_data(self):
            smoke_data = [dict(group=self.matrix_se3, n=3, n_samples=20)]
            return self.generate_tests(smoke_data)

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=1e-2,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=1e-2,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100000,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 1000,
                atol=1e-1,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                amplitude=1000,
                rtol=gs.rtol * 1000,
                atol=1e-1,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

        def exp_log_composition_at_identity_test_data(self):
            smoke_data = []
            for metric_args in self.metric_args_list[:4]:
                for tangent_vec in [self.point_1, self.point_small]:
                    smoke_data += [
                        dict(metric_args=metric_args, tangent_vec=tangent_vec)
                    ]
            return self.generate_tests(smoke_data)

        def log_exp_composition_at_identity_test_data(self):
            smoke_data = []
            for metric_args in self.metric_args_list[:4]:
                for point in [self.point_1, self.point_small]:
                    smoke_data += [dict(metric_args=metric_args, point=point)]
            return self.generate_tests(smoke_data)

        def left_exp_and_exp_from_identity_left_diag_metrics_test_data(self):
            smoke_data = [
                dict(metric_args=self.metric_args_list[0], point=self.point_1)
            ]
            return self.generate_tests(smoke_data)

        def left_log_and_log_from_identity_left_diag_metrics_test_data(self):
            smoke_data = [
                dict(metric_args=self.metric_args_list[0], point=self.point_1)
            ]
            return self.generate_tests(smoke_data)

    testing_data = InvariantMetricTestData()

    def test_inner_product_mat_at_identity_shape(
        self, group, metric_mat_at_identity, left_or_right
    ):
        metric = self.metric(group, metric_mat_at_identity, left_or_right)
        dim = metric.group.dim
        result = metric.metric_mat_at_identity
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_shape(
        self, group, metric_mat_at_identity, left_or_right, base_point
    ):
        metric = self.metric(group, metric_mat_at_identity, left_or_right)
        base_point = None
        dim = metric.group.dim
        result = metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

        base_point = group.identity
        dim = metric.group.dim
        result = metric.metric_matrix(base_point=base_point)
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
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        metric = InvariantMetric(group=group)
        result = metric.structure_constant(tangent_vec_a, tangent_vec_b, tangent_vec_c)
        self.assertAllClose(result, expected)

    def test_dual_adjoint_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c
    ):
        metric = InvariantMetric(group=group)
        result = metric.inner_product_at_identity(
            metric.dual_adjoint(tangent_vec_a, tangent_vec_b), tangent_vec_c
        )
        expected = metric.structure_constant(
            tangent_vec_a, tangent_vec_c, tangent_vec_b
        )
        self.assertAllClose(result, expected)

    def test_connection(self, group, tangent_vec_a, tangent_vec_b, expected):
        metric = InvariantMetric(group)
        self.assertAllClose(metric.connection(tangent_vec_a, tangent_vec_b), expected)

    def test_connection_translation_map(
        self, group, tangent_vec_a, tangent_vec_b, point, expected
    ):
        metric = InvariantMetric(group)
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        result = metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected, rtol=1e-3, atol=1e-3)

    def test_sectional_curvature(self, group, tangent_vec_a, tangent_vec_b, expected):
        metric = InvariantMetric(group)
        result = metric.sectional_curvature(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(result, expected)

    def test_sectional_curvature_translation_point(
        self, group, tangent_vec_a, tangent_vec_b, point, expected
    ):
        metric = InvariantMetric(group)
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        result = metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def test_curvature(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        metric = InvariantMetric(group)
        result = metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point=None
        )
        self.assertAllClose(result, expected)

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

    def test_integrated_exp_at_id(
        self,
        group,
    ):
        metric = InvariantMetric(group=group)
        basis = metric.normal_basis(group.lie_algebra.basis)

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity
        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        expected = group.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-4)

        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk2")
        self.assertAllClose(expected, result, atol=1e-4)

    def test_integrated_se3_exp_at_id(self, group):
        lie_algebra = group.lie_algebra
        metric = InvariantMetric(group=group)
        canonical_metric = group.left_canonical_metric
        basis = metric.normal_basis(lie_algebra.basis)

        vector = gs.random.rand(len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity
        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        expected = canonical_metric.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-4)

        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk2")
        self.assertAllClose(expected, result, atol=1e-4)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_integrated_exp_and_log_at_id(self, group):
        metric = InvariantMetric(group=group)
        basis = group.lie_algebra.basis

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity

        exp = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        result = metric.log(exp, identity, n_steps=15, step="rk4", verbose=False)
        self.assertAllClose(tangent_vec, result, atol=1e-5)

    def test_integrated_parallel_transport(self, group, n, n_samples):
        metric = InvariantMetric(group=group)
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

        self.assertAllClose(end_point_result, expected_end_point, atol=gs.atol * 1000)
        self.assertAllClose(expected, result, atol=gs.atol * 1000)

    def test_log_antipodals(self, group, rotation_mat1, rotation_mat2, expected):
        with expected:
            group.bi_invariant_metric.log(rotation_mat1, rotation_mat2)

    @geomstats.tests.np_autograd_and_tf_only
    def test_left_exp_and_exp_from_identity_left_diag_metrics(self, metric_args, point):
        metric = self.metric(*metric_args)
        left_exp_from_id = metric.left_exp_from_identity(point)
        exp_from_id = metric.exp_from_identity(point)

        self.assertAllClose(left_exp_from_id, exp_from_id)

    @geomstats.tests.np_autograd_and_tf_only
    def test_left_log_and_log_from_identity_left_diag_metrics(self, metric_args, point):
        metric = self.metric(*metric_args)
        left_log_from_id = metric.left_log_from_identity(point)
        log_from_id = metric.log_from_identity(point)

        self.assertAllClose(left_log_from_id, log_from_id)

    @geomstats.tests.np_autograd_and_tf_only
    def test_exp_log_composition_at_identity(self, metric_args, tangent_vec):
        metric = self.metric(*metric_args)
        result = metric.left_log_from_identity(
            point=metric.left_exp_from_identity(tangent_vec=tangent_vec)
        )
        self.assertAllClose(result, tangent_vec)

    @geomstats.tests.np_autograd_and_tf_only
    def test_log_exp_composition_at_identity(self, metric_args, point):
        metric = self.metric(*metric_args)
        result = metric.left_exp_from_identity(
            tangent_vec=metric.left_log_from_identity(point=point)
        )
        self.assertAllClose(result, point)
