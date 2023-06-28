import itertools

import pytest

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import _RiemannianMetricTestData


class InvariantMetricTestData(_RiemannianMetricTestData):
    group = SpecialEuclidean(n=3, point_type="vector")
    matrix_se3 = SpecialEuclidean(n=3, equip=False)
    matrix_so3 = SpecialOrthogonal(n=3, equip=False)
    vector_so3 = SpecialOrthogonal(n=3, point_type="vector", equip=False)
    point_1 = gs.array([-0.2, 0.9, 0.5, 5.0, 5.0, 5.0])
    point_2 = gs.array([0.0, 2.0, -0.1, 30.0, 400.0, 2.0])
    point_1_matrix = vector_so3.matrix_from_rotation_vector(point_1[..., :3])
    point_2_matrix = vector_so3.matrix_from_rotation_vector(point_2[..., :3])
    # Edge case for the point, angle < epsilon,
    point_small = gs.array([-1e-7, 0.0, -7 * 1e-8, 6.0, 5.0, 9.0])

    diag_mat_at_identity = gs.eye(group.dim)
    metric_args_list = [
        dict(metric_mat_at_identity=None, left=True),
        dict(metric_mat_at_identity=None, left=False),
        dict(metric_mat_at_identity=gs.eye(group.dim), left=True),
        dict(metric_mat_at_identity=gs.eye(group.dim), left=False),
        dict(metric_mat_at_identity=None, left=False),
        dict(metric_mat_at_identity=None, left=True),
    ]
    space_list = [group, group, group, group, matrix_so3, matrix_so3]
    shape_list = [space.shape for space in space_list]

    n_points_list = [1, 2] * 3
    n_tangent_vecs_list = [1, 2] * 3
    n_points_a_list = [1, 2] * 3
    n_points_b_list = [1]
    alpha_list = [1] * 6
    n_rungs_list = [1] * 6
    scheme_list = ["pole"] * 6

    Metric = InvariantMetric

    tolerances = {
        "exp_belongs": {"atol": 1e-4},
        "exp_after_log": {"atol": 1e-1},
        "integrated_parallel_transport": {"atol": 1e-8},
        "dist_point_to_itself_is_zero": {"atol": 1e-6},
    }

    def inner_product_mat_at_identity_shape_test_data(self):
        group = SpecialEuclidean(n=3, point_type="vector")
        sym_mat_at_identity = gs.eye(group.dim)
        smoke_data = [
            dict(
                group=group,
                metric_mat_at_identity=sym_mat_at_identity,
                left=True,
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
                left=True,
                base_point=None,
            ),
            dict(
                group=group,
                metric_mat_at_identity=sym_mat_at_identity,
                left=True,
                base_point=group.identity,
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_matrix_and_its_inverse_test_data(self):
        group = SpecialEuclidean(n=3, point_type="vector")
        smoke_data = [dict(group=group, metric_mat_at_identity=None, left=True)]
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
                left=True,
                tangent_vec_a=tangent_vec_a,
                tangent_vec_b=tangent_vec_b,
                base_point=None,
                expected=4.0,
            ),
            dict(
                group=group,
                metric_mat_at_identity=None,
                left=True,
                tangent_vec_a=batch_tangent_vec,
                tangent_vec_b=tangent_vec_b,
                base_point=None,
                expected=gs.array([4.0, 5.0]),
            ),
            dict(
                group=group,
                metric_mat_at_identity=None,
                left=True,
                tangent_vec_a=group.compose(self.point_1_matrix, tangent_vec_a),
                tangent_vec_b=group.compose(self.point_1_matrix, tangent_vec_b),
                base_point=self.point_1_matrix,
                expected=4.0,
            ),
            dict(
                group=group,
                metric_mat_at_identity=None,
                left=True,
                tangent_vec_a=group.compose(self.point_1_matrix, batch_tangent_vec),
                tangent_vec_b=group.compose(self.point_1_matrix, tangent_vec_b),
                base_point=self.point_1_matrix,
                expected=gs.array([4.0, 5.0]),
            ),
            dict(
                group=group,
                metric_mat_at_identity=None,
                left=False,
                tangent_vec_a=group.compose(tangent_vec_a, self.point_1_matrix),
                tangent_vec_b=group.compose(tangent_vec_b, self.point_1_matrix),
                base_point=self.point_1_matrix,
                expected=4.0,
            ),
            dict(
                group=group,
                metric_mat_at_identity=None,
                left=False,
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
        metric = InvariantMetric(group)
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

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=1000.0)

    def exp_log_composition_at_identity_test_data(self):
        smoke_data = []
        for group, metric_args in zip(self.space_list[:4], self.metric_args_list[:4]):
            for tangent_vec in [self.point_1, self.point_small]:
                smoke_data += [
                    dict(group=group, metric_args=metric_args, tangent_vec=tangent_vec)
                ]
        return self.generate_tests(smoke_data)

    def log_exp_composition_at_identity_test_data(self):
        smoke_data = []
        for group, metric_args in zip(self.space_list[:4], self.metric_args_list[:4]):
            for point in [self.point_1, self.point_small]:
                smoke_data += [dict(group=group, metric_args=metric_args, point=point)]
        return self.generate_tests(smoke_data)

    def left_exp_and_exp_from_identity_left_diag_metrics_test_data(self):
        smoke_data = [
            dict(
                group=self.space_list[0],
                metric_args=self.metric_args_list[0],
                point=self.point_1,
            )
        ]
        return self.generate_tests(smoke_data)

    def left_log_and_log_from_identity_left_diag_metrics_test_data(self):
        smoke_data = [
            dict(
                group=self.space_list[0],
                metric_args=self.metric_args_list[0],
                point=self.point_1,
            )
        ]
        return self.generate_tests(smoke_data)
