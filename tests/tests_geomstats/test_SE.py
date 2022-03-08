"""Unit tests for special euclidean group in matrix representation."""

import itertools
import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatrixCannonicalLeftMetric,
    SpecialEuclideanMatrixLieAlgebra,
)
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.tests import tf_backend
from tests.conftest import TestCase, np_backend
from tests.data_generation import (
    LieGroupTestData,
    MatrixLieAlgebraTestData,
    RiemannianMetricTestData,
)
from tests.parametrizers import (
    LieGroupParametrizer,
    MatrixLieAlgebraParametrizer,
    RiemannianMetricParametrizer,
)


def group_useful_matrix(theta, elem_33=1.0):
    return gs.array(
        [
            [gs.cos(theta), -gs.sin(theta), 2.0],
            [gs.sin(theta), gs.cos(theta), 3.0],
            [0.0, 0.0, elem_33],
        ]
    )


def algebra_useful_matrix(theta, elem_33=0.0):
    return gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, elem_33]])


point_1 = gs.array([0.1, 0.2, 0.3])
point_2 = gs.array([0.5, 5.0, 60.0])

translation_large = gs.array([0.0, 5.0, 6.0])
translation_small = gs.array([0.0, 0.6, 0.7])

elements_all = {
    "translation_large": translation_large,
    "translation_small": translation_small,
    "point_1": point_1,
    "point_2": point_2,
}
elements = elements_all
if tf_backend():
    # Tf is extremely slow
    elements = {"point_1": point_1, "point_2": point_2}

elements_matrices_all = {
    key: SpecialEuclidean(2, point_type="vector").matrix_from_vector(elements_all[key])
    for key in elements_all
}
elements_matrices = elements_matrices_all


class TestSpecialEuclidean(TestCase, metaclass=LieGroupParametrizer):

    space = group = SpecialEuclidean

    class TestDataSpecialEuclidean(LieGroupTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list] + [(2, "vector"), (3, "vector")]
        shape_list = [(n + 1, n + 1) for n in n_list]
        n_samples_list = random.sample(range(2, 10), 4)
        n_points_list = random.sample(range(2, 10), 4)
        n_vecs_list = random.sample(range(2, 10), 4)

        def belongs_data(self):
            smoke_data = [
                dict(
                    n=2, mat=group_useful_matrix(gs.pi / 3, elem_33=1.0), expected=True
                ),
                dict(
                    n=2, mat=group_useful_matrix(gs.pi / 3, elem_33=0.0), expected=False
                ),
                dict(
                    n=2,
                    mat=[
                        group_useful_matrix(gs.pi / 3, elem_33=1.0),
                        group_useful_matrix(gs.pi / 3, elem_33=0.0),
                    ],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def identity_data(self):
            smoke_data = [
                dict(n=2, expected=gs.eye(3)),
                dict(n=3, expected=gs.eye(4)),
                dict(n=10, expected=gs.eye(11)),
            ]
            return self.generate_tests(smoke_data)

        def is_tangent_data(self):
            theta = gs.pi / 3
            vec_1 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 0.0]])
            vec_2 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 0.0]])
            point = group_useful_matrix(theta)
            smoke_data = [
                dict(n=2, tangent_vec=point @ vec_1, base_point=point, expected=True),
                dict(n=2, tangent_vec=point @ vec_2, base_point=point, expected=False),
                dict(
                    n=2,
                    tangent_vec=[point @ vec_1, point @ vec_2],
                    base_point=point,
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def compose_identity_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 100
            random_data = [
                dict(n=n, point=SpecialEuclidean(n).random_point(n_samples))
                for n in n_list
            ]
            return self.generate_tests([], random_data)

        def basis_representation_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 100
            random_data = [
                dict(n=n, vec=gs.random.rand(n_samples, self.group.dim)) for n in n_list
            ]
            return self.generate_tests([], random_data)

        def metrics_default_point_type_data(self):
            n_list = random.sample(range(2, 5), 2)
            metric_str_list = [
                "left_canonical_metric",
                "right_canonical_metric",
                "metric",
            ]
            random_data = [arg for arg in itertools.product(n_list, metric_str_list)]
            return self.generate_tests([], random_data)

        def inverse_shape_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 10
            random_data = [
                dict(
                    n=n,
                    points=SpecialEuclidean(n).random_point(n_samples),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            return self.generate_tests([], random_data)

        def compose_shape_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 10
            random_data = [
                dict(
                    n=n,
                    point_a=SpecialEuclidean(n).random_point(n_samples),
                    point_b=SpecialEuclidean(n).random_point(n_samples),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            random_data += [
                dict(
                    n=n,
                    point_a=SpecialEuclidean(n).random_point(),
                    point_b=SpecialEuclidean(n).random_point(n_samples),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            random_data += [
                dict(
                    n=n,
                    point_a=SpecialEuclidean(n).random_point(n_samples),
                    point_b=SpecialEuclidean(n).random_point(),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            return self.generate_tests([], random_data)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2, True), (3, True), (2, False)]
            smoke_n_points_list = [1, 2, 1]
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                SpecialEuclidean,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                SpecialEuclidean,
                self.space_args_list,
                self.shape_list,
                self.n_samples_list,
                atol=gs.atol * 1000,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                SpecialEuclidean,
                self.space_args_list,
                self.n_samples_list,
                atol=gs.atol * 1000,
            )

        def regularize_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_type="vector",
                    point=elements_all["point_1"],
                    expected=elements_all["point_1"],
                )
            ]
            return self.generate_tests(smoke_data)

        def regularize_shape_data(self):
            smoke_data = [dict(n=2, point_type="vector", n_samples=3)]
            return self.generate_tests(smoke_data)

        # def test_compose_point_identity_is_point(self):
        #     smoke_data = [dict(n)]
        #     return self.generate_tests(smoke_data)

    testing_data = TestDataSpecialEuclidean()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(
            SpecialEuclidean(n).belongs(gs.array(mat)), gs.array(expected)
        )

    def test_random_point_belongs(self, n, n_samples):
        group = self.cls(n)
        self.assertAllClose(gs.all(group(n).random_point(n_samples)), gs.array(True))

    def test_identity(self, n, expected):
        self.assertAllClose(SpecialEuclidean(n).identity, gs.array(expected))

    def test_is_tangent(self, n, tangent_vec, base_point, expected):
        result = SpecialEuclidean(n).is_tangent(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_compose_identity(self, n, point):
        group = self.space(n)
        result = group.compose(gs.array(point), group.inverse(gs.array(point)))
        self.assertAllClose(result, gs.broadcast_to(group.identity, result.shape))

    # def test_basis_representation(self, n, point_type, vec):
    #     group = self.cls(n, point_type)
    #     tangent_vec = group.lie_algebra.matrix_representation(vec)
    #     result = group.lie_algebra.basis_representation(tangent_vec)
    #     self.assertAllClose(result, vec)

    def test_metrics_default_point_type(self, n, metric_str):
        group = self.space(n)
        self.assertTrue(getattr(group, metric_str).default_point_type == "matrix")

    def test_inverse_shape(self, n, points, expected):
        group = self.space(n)
        self.assertAllClose(gs.shape(group.inverse(points)), expected)

    def test_compose_shape(self, n, point_a, point_b, expected):
        group = self.space(n)
        result = gs.shape(group.compose(gs.array(point_a), gs.array(point_b)))
        self.assertAllClose(result, expected)

    def test_regularize_shape(self, n, point_type, n_samples):
        group = self.space(n, point_type)
        points = group.random_point(n_samples=n_samples)
        regularized_points = group.regularize(points)

        self.assertAllClose(
            gs.shape(regularized_points),
            (n_samples, *group.get_point_type_shape()),
        )


class TestSpecialEuclideanMatrixLieAlgebra(
    TestCase, metaclass=MatrixLieAlgebraParametrizer
):
    space = algebra = SpecialEuclideanMatrixLieAlgebra
    skip_test_basis_representation_matrix_representation_composition = True
    skip_test_matrix_representation_basis_representation_composition = True
    skip_test_random_point_belongs = True

    class TestDataSpecialEuclideanMatrixLieAlgebra(MatrixLieAlgebraTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n + 1, n + 1) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_data(self):
            theta = gs.pi / 3
            smoke_data = [
                dict(n=2, vec=algebra_useful_matrix(theta, elem_33=0.0), expected=True),
                dict(
                    n=2, vec=algebra_useful_matrix(theta, elem_33=1.0), expected=False
                ),
                dict(
                    n=2,
                    vec=[
                        algebra_useful_matrix(theta, elem_33=0.0),
                        algebra_useful_matrix(theta, elem_33=1.0),
                    ],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def dim_data(self):
            smoke_data = [
                dict(n=2, expected=3),
                dict(n=3, expected=6),
                dict(n=10, expected=55),
            ]
            return self.generate_tests(smoke_data)

        def basis_representation_matrix_representation_composition_data(self):
            return self._basis_representation_matrix_representation_composition_data(
                SpecialEuclideanMatrixLieAlgebra,
                self.space_args_list,
                self.n_samples_list,
            )

        def matrix_representation_basis_representation_composition_data(self):
            return self._matrix_representation_basis_representation_composition_data(
                SpecialEuclideanMatrixLieAlgebra,
                self.space_args_list,
                self.n_samples_list,
            )

        def basis_belongs_data(self):
            return self._basis_belongs_data(self.space_args_list)

        def basis_cardinality_data(self):
            return self._basis_cardinality_data(self.space_args_list)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                SpecialEuclideanMatrixLieAlgebra,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

    testing_data = TestDataSpecialEuclideanMatrixLieAlgebra()

    def test_dim(self, n, expected):
        algebra = self.space(n)
        self.assertAllClose(algebra.dim, expected)

    def test_belongs(self, n, vec, expected):
        algebra = self.space(n)
        self.assertAllClose(algebra.belongs(gs.array(vec)), gs.array(expected))


class TestSpecialEuclideanMatrixCannonicalLeftMetric(
    TestCase,
    metaclass=RiemannianMetricParametrizer,
):

    metric = connection = SpecialEuclideanMatrixCannonicalLeftMetric
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = True

    class TestDataSpecialEuclideanMatrixCanonicalLeftMetric(RiemannianMetricTestData):
        n_list = random.sample(range(2, 5), 2)
        metric_args_list = [(SpecialEuclidean(n),) for n in n_list]
        shape_list = [(n + 1, n + 1) for n in n_list]
        space_list = [SpecialEuclidean(n) for n in n_list]
        n_points_list = random.sample(range(1, 7), 2)
        n_samples_list = random.sample(range(1, 7), 2)
        n_points_a_list = random.sample(range(1, 7), 2)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 7), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def left_metric_wrong_group_data(self):
            smoke_data = [
                dict(group=SpecialEuclidean(2), expected=does_not_raise()),
                dict(group=SpecialEuclidean(3), expected=does_not_raise()),
                dict(
                    group=SpecialEuclidean(2, point_type="vector"),
                    expected=pytest.raises(ValueError),
                ),
                dict(group=SpecialOrthogonal(3), expected=pytest.raises(ValueError)),
            ]
            return self.generate_tests(smoke_data)

        def exp_shape_data(self):
            return self._exp_shape_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.batch_size_list,
            )

        def log_shape_data(self):
            return self._log_shape_data(
                self.metric_args_list,
                self.space_list,
                self.batch_size_list,
            )

        def squared_dist_is_symmetric_data(self):
            return self._squared_dist_is_symmetric_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_data(self):
            return self._exp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_data(self):
            return self._log_is_tangent_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_data(self):
            return self._geodesic_ivp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100,
            )

        def geodesic_bvp_belongs_data(self):
            return self._geodesic_bvp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def exp_ladder_parallel_transport_data(self):
            return self._exp_ladder_parallel_transport_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_data(self):
            return self._exp_geodesic_ivp_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def parallel_transport_ivp_is_isometry_data(self):
            return self._parallel_transport_ivp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_data(self):
            return self._parallel_transport_bvp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

    testing_data = TestDataSpecialEuclideanMatrixCanonicalLeftMetric()

    def test_left_metric_wrong_group(self, group, expected):
        with expected:
            self.metric(group)


# class TestSpecialEuclideanMatrixCannonicalRightMetric(
#     TestCase,
#     metaclass=RiemannianMetricParametrizer,
# ):

#     metric = connection = InvariantMetric
#     skip_test_exp_geodesic_ivp = True
#     skip_test_exp_shape = np_backend()
#     skip_test_log_shape = np_backend()
#     skip_test_parallel_transport_ivp_is_isometry = True
#     skip_test_parallel_transport_bvp_is_isometry = True
#     skip_test_squared_dist_is_symmetric = True
#     skip_test_exp_log_composition = True
#     skip_test_log_exp_composition = True

#     class TestDataSpecialEuclideanMatrixCanonicalRightMetric(RiemannianMetricTestData):
#         n_list = random.sample(range(2, 4), 2)
#         metric_args_list = [
#             (SpecialEuclidean(n), gs.eye(SpecialEuclidean(n).dim), "right")
#             for n in n_list
#         ]
#         shape_list = [(n + 1, n + 1) for n in n_list]
#         space_list = [SpecialEuclidean(n) for n in n_list]
#         n_points_list = random.sample(range(1, 7), 2)
#         n_samples_list = random.sample(range(1, 7), 2)
#         n_points_a_list = random.sample(range(1, 7), 2)
#         n_points_b_list = [1]
#         batch_size_list = random.sample(range(2, 7), 2)
#         alpha_list = [1] * 2
#         n_rungs_list = [1] * 2
#         scheme_list = ["pole"] * 2

#         def exp_shape_data(self):
#             return self._exp_shape_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.batch_size_list,
#             )

#         def log_shape_data(self):
#             return self._log_shape_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.batch_size_list,
#             )

#         def squared_dist_is_symmetric_data(self):
#             return self._squared_dist_is_symmetric_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.n_points_a_list,
#                 self.n_points_b_list,
#                 atol=gs.atol * 1000,
#             )

#         def exp_belongs_data(self):
#             return self._exp_belongs_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.n_samples_list,
#                 belongs_atol=gs.atol * 1000,
#             )

#         def log_is_tangent_data(self):
#             return self._log_is_tangent_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.n_samples_list,
#                 is_tangent_atol=gs.atol * 1000,
#             )

#         def geodesic_ivp_belongs_data(self):
#             return self._geodesic_ivp_belongs_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.n_points_list,
#                 belongs_atol=gs.atol * 100,
#             )

#         def geodesic_bvp_belongs_data(self):
#             return self._geodesic_bvp_belongs_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.n_points_list,
#                 belongs_atol=gs.atol * 100,
#             )

#         def log_exp_composition_data(self):
#             return self._log_exp_composition_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.n_samples_list,
#                 rtol=gs.rtol * 10000,
#                 atol=gs.atol * 10000,
#             )

#         def exp_log_composition_data(self):
#             return self._exp_log_composition_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.n_samples_list,
#                 amplitude=10.0,
#                 rtol=gs.rtol * 10000,
#                 atol=gs.atol * 10000,
#             )

#         def exp_ladder_parallel_transport_data(self):
#             return self._exp_ladder_parallel_transport_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.n_samples_list,
#                 self.n_rungs_list,
#                 self.alpha_list,
#                 self.scheme_list,
#             )

#         def exp_geodesic_ivp_data(self):
#             return self._exp_geodesic_ivp_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.n_samples_list,
#                 self.n_points_list,
#                 rtol=gs.rtol * 100,
#                 atol=gs.atol * 100,
#             )

#         def parallel_transport_ivp_is_isometry_data(self):
#             return self._parallel_transport_ivp_is_isometry_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.n_samples_list,
#                 is_tangent_atol=gs.atol * 1000,
#                 atol=gs.atol * 1000,
#             )

#         def parallel_transport_bvp_is_isometry_data(self):
#             return self._parallel_transport_bvp_is_isometry_data(
#                 self.metric_args_list,
#                 self.space_list,
#                 self.shape_list,
#                 self.n_samples_list,
#                 is_tangent_atol=gs.atol * 1000,
#                 atol=gs.atol * 1000,
#             )

#         def right_exp_coincides_data(self):
#             smoke_data = [
#                 dict(
#                     n=2,
#                     point_type="vector",
#                     initial_vec=gs.array([gs.pi / 2, 1.0, 1.0]),
#                 )
#             ]
#             return self.generate_tests(smoke_data)

#     testing_data = TestDataSpecialEuclideanMatrixCanonicalRightMetric()

#     def test_right_exp_coincides(self, initial_vec):
#         vector_group = SpecialEuclidean(n=2, point_type="vector")
#         initial_matrix_vec = self.group.lie_algebra.matrix_representation(initial_vec)
#         vector_exp = vector_group.right_canonical_metric.exp(initial_vec)
#         result = self.group.right_canonical_metric.exp(initial_matrix_vec, n_steps=25)
#         expected = vector_group.matrix_from_vector(vector_exp)
#         self.assertAllClose(result, expected, atol=1e-6)
