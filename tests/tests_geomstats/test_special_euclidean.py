"""Unit tests for special euclidean group in matrix representation."""

import itertools
import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatrixCannonicalLeftMetric,
    SpecialEuclideanMatrixLieAlgebra,
)
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.tests import tf_backend
from tests.conftest import Parametrizer, TestCase, np_backend
from tests.data_generation import (
    TestData,
    _LieGroupTestData,
    _MatrixLieAlgebraTestData,
    _RiemannianMetricTestData,
)
from tests.geometry_test_cases import (
    LieGroupTestCase,
    MatrixLieAlgebraTestCase,
    RiemannianMetricTestCase,
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

ATOL = 1e-5


class TestSpecialEuclidean(LieGroupTestCase, metaclass=Parametrizer):

    space = group = SpecialEuclidean
    skip_test_exp_then_log = tf_backend()
    skip_test_log_then_exp = tf_backend()

    class SpecialEuclideanTestData(_LieGroupTestData):
        n_list = random.sample(range(2, 4), 2)
        space_args_list = [(n,) for n in n_list] + [(2, "vector"), (3, "vector")]
        shape_list = [(n + 1, n + 1) for n in n_list] + [(3,)] + [(6,)]
        n_tangent_vecs_list = [2, 3] * 2
        n_points_list = [2, 3] * 2
        n_vecs_list = [2, 3] * 2

        def belongs_test_data(self):
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

        def identity_test_data(self):
            smoke_data = [
                dict(n=2, expected=gs.eye(3)),
                dict(n=3, expected=gs.eye(4)),
                dict(n=10, expected=gs.eye(11)),
            ]
            return self.generate_tests(smoke_data)

        def is_tangent_test_data(self):
            theta = gs.pi / 3
            vec_1 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 0.0]])
            vec_2 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 1.0]])
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

        def basis_representation_test_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 100
            random_data = [
                dict(n=n, vec=gs.random.rand(n_samples, self.group.dim)) for n in n_list
            ]
            return self.generate_tests([], random_data)

        def metrics_default_point_type_test_data(self):
            n_list = random.sample(range(2, 5), 2)
            metric_str_list = [
                "left_canonical_metric",
                "right_canonical_metric",
                "metric",
            ]
            random_data = itertools.product(n_list, metric_str_list)
            return self.generate_tests([], random_data)

        def inverse_shape_test_data(self):
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

        def compose_shape_test_data(self):
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

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2, True), (3, True), (2, False)]
            smoke_n_points_list = [1, 2, 1]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=1e-2,
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                SpecialEuclidean,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                SpecialEuclidean,
                self.space_args_list,
                self.n_vecs_list,
                is_tangent_atol=gs.atol * 100,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                SpecialEuclidean,
                self.space_args_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                amplitude=100,
                atol=gs.atol * 10000,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                SpecialEuclidean,
                self.space_args_list,
                self.n_points_list,
                atol=gs.atol * 10000,
            )

        def regularize_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_type="vector",
                    point=elements_all["point_1"],
                    expected=elements_all["point_1"],
                )
            ]
            return self.generate_tests(smoke_data)

        def regularize_shape_test_data(self):
            smoke_data = [dict(n=2, point_type="vector", n_samples=3)]
            return self.generate_tests(smoke_data)

        def compose_inverse_point_with_point_is_identity_test_data(self):
            return self._compose_inverse_point_with_point_is_identity_test_data(
                SpecialEuclidean, self.space_args_list, self.n_points_list
            )

        def compose_point_with_inverse_point_is_identity_test_data(self):
            return self._compose_point_with_inverse_point_is_identity_test_data(
                SpecialEuclidean, self.space_args_list, self.n_points_list
            )

        def compose_point_with_identity_is_point_test_data(self):
            return self._compose_point_with_identity_is_point_test_data(
                SpecialEuclidean, self.space_args_list, self.n_points_list
            )

        def compose_identity_with_point_is_point_test_data(self):
            return self._compose_identity_with_point_is_point_test_data(
                SpecialEuclidean, self.space_args_list, self.n_points_list
            )

        def compose_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_typ="vector",
                    point_1=elements_all["translation_small"],
                    point_2=elements_all["translation_large"],
                    expected=elements_all["translation_small"]
                    + elements_all["translation_large"],
                )
            ]
            return self.generate_tests(smoke_data)

        def group_exp_from_identity_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_type="vector",
                    tangent_vec=elements_all["translation_small"],
                    expected=elements_all["translation_small"],
                ),
                dict(
                    n=2,
                    point_type="vector",
                    tangent_vec=gs.stack([elements_all["translation_small"]] * 2),
                    expected=gs.stack([elements_all["translation_small"]] * 2),
                ),
            ]
            return self.generate_tests(smoke_data)

        def group_log_from_identity_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_type="vector",
                    point=elements_all["translation_small"],
                    expected=elements_all["translation_small"],
                ),
                dict(
                    n=2,
                    point_type="vector",
                    point=gs.stack([elements_all["translation_small"]] * 2),
                    expected=gs.stack([elements_all["translation_small"]] * 2),
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = SpecialEuclideanTestData()

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

    def test_compose(self, n, point_type, point_1, point_2, expected):
        group = self.space(n, point_type)
        result = group.compose(point_1, point_2)
        self.assertAllClose(result, expected)

    def test_group_exp_from_identity(self, n, point_type, tangent_vec, expected):
        group = self.space(n, point_type)
        result = group.exp(base_point=group.identity, tangent_vec=tangent_vec)
        self.assertAllClose(result, expected)

    def test_group_log_from_identity(self, n, point_type, point, expected):
        group = self.space(n, point_type)
        result = group.log(base_point=group.identity, point=point)
        self.assertAllClose(result, expected)


class TestSpecialEuclideanMatrixLieAlgebra(
    MatrixLieAlgebraTestCase, metaclass=Parametrizer
):
    space = algebra = SpecialEuclideanMatrixLieAlgebra

    class SpecialEuclideanMatrixLieAlgebraTestData(_MatrixLieAlgebraTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n + 1, n + 1) for n in n_list]
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
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

        def dim_test_data(self):
            smoke_data = [
                dict(n=2, expected=3),
                dict(n=3, expected=6),
                dict(n=10, expected=55),
            ]
            return self.generate_tests(smoke_data)

        def basis_representation_then_matrix_representation_test_data(self):
            return self._basis_representation_then_matrix_representation_test_data(
                SpecialEuclideanMatrixLieAlgebra,
                self.space_args_list,
                self.n_points_list,
            )

        def matrix_representation_then_basis_representation_test_data(self):
            return self._matrix_representation_then_basis_representation_test_data(
                SpecialEuclideanMatrixLieAlgebra,
                self.space_args_list,
                self.n_points_list,
            )

        def basis_belongs_test_data(self):
            return self._basis_belongs_test_data(self.space_args_list)

        def basis_cardinality_test_data(self):
            return self._basis_cardinality_test_data(self.space_args_list)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                SpecialEuclideanMatrixLieAlgebra,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                SpecialEuclideanMatrixLieAlgebra, self.space_args_list, self.n_vecs_list
            )

        def to_tangent_is_projection_test_data(self):
            return self._to_tangent_is_projection_test_data(
                SpecialEuclideanMatrixLieAlgebra,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_point_is_tangent_test_data(self):
            return self._random_point_is_tangent_test_data(
                self.space_args_list, self.n_points_list
            )

    testing_data = SpecialEuclideanMatrixLieAlgebraTestData()

    def test_dim(self, n, expected):
        algebra = self.space(n)
        self.assertAllClose(algebra.dim, expected)

    def test_belongs(self, n, vec, expected):
        algebra = self.space(n)
        self.assertAllClose(algebra.belongs(gs.array(vec)), gs.array(expected))


class TestSpecialEuclideanMatrixCanonicalLeftMetric(
    RiemannianMetricTestCase,
    metaclass=Parametrizer,
):

    metric = connection = SpecialEuclideanMatrixCannonicalLeftMetric
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = True

    class SpecialEuclideanMatrixCanonicalLeftMetricTestData(_RiemannianMetricTestData):
        n_list = random.sample(range(2, 5), 2)
        metric_args_list = [(SpecialEuclidean(n),) for n in n_list]
        shape_list = [(n + 1, n + 1) for n in n_list]
        space_list = [SpecialEuclidean(n) for n in n_list]
        n_points_list = [2, 3]
        n_tangent_vecs_list = [2, 3]
        n_points_a_list = [2, 3]
        n_points_b_list = [1]
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def left_metric_wrong_group_test_data(self):
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

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(
                self.metric_args_list,
                self.space_list,
            )

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
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 100,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                amplitude=10,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
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
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
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

    testing_data = SpecialEuclideanMatrixCanonicalLeftMetricTestData()

    def test_left_metric_wrong_group(self, group, expected):
        with expected:
            self.metric(group)


class TestSpecialEuclideanMatrixCanonicalRightMetric(
    RiemannianMetricTestCase,
    metaclass=Parametrizer,
):

    metric = connection = InvariantMetric
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = np_backend()
    skip_test_log_shape = np_backend()
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_squared_dist_is_symmetric = np_backend()
    skip_test_exp_then_log = True
    skip_test_log_then_exp = True
    skip_test_log_is_tangent = np_backend()
    skip_test_geodesic_bvp_belongs = np_backend()
    skip_test_exp_ladder_parallel_transport = np_backend()
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_belongs = np_backend()
    skip_test_squared_dist_is_symmetric = True
    skip_test_dist_is_norm_of_log = True
    skip_test_dist_is_positive = np_backend()
    skip_test_dist_is_symmetric = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_squared_dist_is_positive = np_backend()

    class SpecialEuclideanMatrixCanonicalRightMetricTestData(_RiemannianMetricTestData):
        n_list = [2]
        metric_args_list = [
            (SpecialEuclidean(n), gs.eye(SpecialEuclidean(n).dim), "right")
            for n in n_list
        ]
        shape_list = [(n + 1, n + 1) for n in n_list]
        space_list = [SpecialEuclidean(n) for n in n_list]
        n_points_list = random.sample(range(1, 3), 1)
        n_tangent_vecs_list = random.sample(range(1, 3), 1)
        n_points_a_list = random.sample(range(1, 3), 1)
        n_points_b_list = [1]
        alpha_list = [1] * 1
        n_rungs_list = [1] * 1
        scheme_list = ["pole"] * 1

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(
                self.metric_args_list,
                self.space_list,
            )

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
                belongs_atol=1e-3,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=1e-3,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=1e-3,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                amplitude=100.0,
                rtol=gs.rtol * 10000,
                atol=gs.atol * 100000,
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
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
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

        def right_exp_coincides_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    initial_vec=gs.array([gs.pi / 2, 1.0, 1.0]),
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = SpecialEuclideanMatrixCanonicalRightMetricTestData()

    def test_right_exp_coincides(self, n, initial_vec):
        group = SpecialEuclidean(n=n)
        vector_group = SpecialEuclidean(n=n, point_type="vector")
        initial_matrix_vec = group.lie_algebra.matrix_representation(initial_vec)
        vector_exp = vector_group.right_canonical_metric.exp(initial_vec)
        result = group.right_canonical_metric.exp(initial_matrix_vec, n_steps=25)
        expected = vector_group.matrix_from_vector(vector_exp)
        self.assertAllClose(result, expected, atol=1e-6)


class TestSpecialEuclidean3Vectors(TestCase, metaclass=Parametrizer):
    space = SpecialEuclidean

    class SpecialEuclidean3VectorsTestData(TestData):
        group = SpecialEuclidean(n=3, point_type="vector")
        angle_0 = gs.zeros(6)
        angle_close_0 = 1e-10 * gs.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0]) + gs.array(
            [0.0, 0.0, 0.0, 1.0, 5.0, 2]
        )
        angle_close_pi_low = (gs.pi - 1e-9) / gs.sqrt(2.0) * gs.array(
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        ) + gs.array([0.0, 0.0, 0.0, -100.0, 0.0, 2.0])
        angle_pi = gs.pi / gs.sqrt(3.0) * gs.array(
            [1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        ) + gs.array([0.0, 0.0, 0.0, -10.2, 0.0, 2.6])
        angle_close_pi_high = (gs.pi + 1e-9) / gs.sqrt(3.0) * gs.array(
            [-1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        ) + gs.array([0.0, 0.0, 0.0, -100.0, 0.0, 2.0])
        angle_in_pi_2pi = (gs.pi + 0.3) / gs.sqrt(5.0) * gs.array(
            [-2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ) + gs.array([0.0, 0.0, 0.0, -100.0, 0.0, 2.0])
        angle_close_2pi_low = (2 * gs.pi - 1e-9) / gs.sqrt(6.0) * gs.array(
            [2.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        ) + gs.array([0.0, 0.0, 0.0, 8.0, 555.0, -2.0])
        angle_2pi = 2.0 * gs.pi / gs.sqrt(3.0) * gs.array(
            [1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        ) + gs.array([0.0, 0.0, 0.0, 1.0, 8.0, -10.0])
        angle_close_2pi_high = (2.0 * gs.pi + 1e-9) / gs.sqrt(2.0) * gs.array(
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
        ) + gs.array([0.0, 0.0, 0.0, 1.0, 8.0, -10.0])

        point_1 = gs.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        point_2 = gs.array([0.5, 0.0, -0.3, 0.4, 5.0, 60.0])

        translation_large = gs.array([0.0, 0.0, 0.0, 0.4, 0.5, 0.6])
        translation_small = gs.array([0.0, 0.0, 0.0, 0.5, 0.6, 0.7])
        rot_with_parallel_trans = gs.array([gs.pi / 3.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        elements_all = {
            "angle_0": angle_0,
            "angle_close_0": angle_close_0,
            "angle_close_pi_low": angle_close_pi_low,
            "angle_pi": angle_pi,
            "angle_close_pi_high": angle_close_pi_high,
            "angle_in_pi_2pi": angle_in_pi_2pi,
            "angle_close_2pi_low": angle_close_2pi_low,
            "angle_2pi": angle_2pi,
            "angle_close_2pi_high": angle_close_2pi_high,
            "translation_large": translation_large,
            "translation_small": translation_small,
            "point_1": point_1,
            "point_2": point_2,
            "rot_with_parallel_trans": rot_with_parallel_trans,
        }
        elements = elements_all
        if geomstats.tests.tf_backend():
            # Tf is extremely slow
            elements = {
                "point_1": point_1,
                "point_2": point_2,
                "angle_close_pi_low": angle_close_pi_low,
            }

        # Metrics - only diagonals
        diag_mat_at_identity = gs.eye(6) * gs.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])

        left_diag_metric = InvariantMetric(
            group=group,
            metric_mat_at_identity=diag_mat_at_identity,
            left_or_right="left",
        )
        right_diag_metric = InvariantMetric(
            group=group,
            metric_mat_at_identity=diag_mat_at_identity,
            left_or_right="right",
        )

        metrics_all = {
            "left_canonical": group.left_canonical_metric,
            "right_canonical": group.right_canonical_metric,
            "left_diag": left_diag_metric,
            "right_diag": right_diag_metric,
        }
        # FIXME:
        # 'left': left_metric,
        # 'right': right_metric}
        metrics = metrics_all
        if geomstats.tests.tf_backend():
            metrics = {"left_diag": left_diag_metric}

        angles_close_to_pi_all = [
            "angle_close_pi_low",
            "angle_pi",
            "angle_close_pi_high",
        ]
        angles_close_to_pi = angles_close_to_pi_all
        if geomstats.tests.tf_backend():
            angles_close_to_pi = ["angle_close_pi_low"]

        def log_then_exp_right_with_angles_close_to_pi_test_data(self):
            smoke_data = []
            for metric in list(self.metrics.values()) + [SpecialEuclidean(3, "vector")]:
                for base_point in self.elements.values():
                    for element_type in self.angles_close_to_pi:
                        point = self.elements[element_type]
                        smoke_data.append(
                            dict(
                                metric=metric,
                                point=point,
                                base_point=base_point,
                            )
                        )
            return self.generate_tests(smoke_data)

        def log_then_exp_test_data(self):
            smoke_data = []
            for metric in list(self.metrics.values()) + [SpecialEuclidean(3, "vector")]:
                for base_point in self.elements.values():
                    for element_type in self.elements:
                        if element_type in self.angles_close_to_pi:
                            continue
                        point = self.elements[element_type]
                        smoke_data.append(
                            dict(
                                metric=metric,
                                point=point,
                                base_point=base_point,
                            )
                        )
            return self.generate_tests(smoke_data)

        def exp_then_log_with_angles_close_to_pi_test_data(self):
            smoke_data = []
            for metric in self.metrics_all.values():
                for base_point in self.elements.values():
                    for element_type in self.angles_close_to_pi:
                        tangent_vec = self.elements_all[element_type]
                        smoke_data.append(
                            dict(
                                metric=metric,
                                tangent_vec=tangent_vec,
                                base_point=base_point,
                            )
                        )
            return self.generate_tests(smoke_data)

        def exp_then_log_test_data(self):
            smoke_data = []
            for metric in [
                self.metrics_all["left_canonical"],
                self.metrics_all["left_diag"],
            ]:
                for base_point in self.elements.values():
                    for element_type in self.elements:
                        if element_type in self.angles_close_to_pi:
                            continue
                        tangent_vec = self.elements[element_type]
                        smoke_data.append(
                            dict(
                                metric=metric,
                                tangent_vec=tangent_vec,
                                base_point=base_point,
                            )
                        )
            return self.generate_tests(smoke_data)

        def exp_test_data(self):
            rot_vec_base_point = gs.array([0.0, 0.0, 0.0])
            translation_base_point = gs.array([4.0, -1.0, 10000.0])
            transfo_base_point = gs.concatenate(
                [rot_vec_base_point, translation_base_point], axis=0
            )

            # Tangent vector is a translation (no infinitesimal rotational part)
            # Expect the sum of the translation
            # with the translation of the reference point
            rot_vec = gs.array([0.0, 0.0, 0.0])
            translation = gs.array([1.0, 0.0, -3.0])
            tangent_vec = gs.concatenate([rot_vec, translation], axis=0)
            expected = gs.concatenate(
                [gs.array([0.0, 0.0, 0.0]), gs.array([5.0, -1.0, 9997.0])], axis=0
            )
            smoke_data = [
                dict(
                    metric=self.metrics_all["left_canonical"],
                    tangent_vec=tangent_vec,
                    base_point=transfo_base_point,
                    expected=expected,
                ),
                dict(
                    metric=self.group,
                    tangent_vec=self.elements_all["translation_small"],
                    base_point=self.elements_all["translation_large"],
                    expected=self.elements_all["translation_large"]
                    + self.elements_all["translation_small"],
                ),
            ]
            return self.generate_tests(smoke_data)

        def log_test_data(self):
            rot_vec_base_point = gs.array([0.0, 0.0, 0.0])
            translation_base_point = gs.array([4.0, 0.0, 0.0])
            transfo_base_point = gs.concatenate(
                [rot_vec_base_point, translation_base_point], axis=0
            )

            # Point is a translation (no rotational part)
            # Expect the difference of the translation
            # by the translation of the reference point
            rot_vec = gs.array([0.0, 0.0, 0.0])
            translation = gs.array([-1.0, -1.0, -1.2])
            point = gs.concatenate([rot_vec, translation], axis=0)

            expected = gs.concatenate(
                [gs.array([0.0, 0.0, 0.0]), gs.array([-5.0, -1.0, -1.2])], axis=0
            )
            smoke_data = [
                dict(
                    metric=self.metrics_all["left_canonical"],
                    point=point,
                    base_point=transfo_base_point,
                    expected=expected,
                ),
                dict(
                    metric=self.group,
                    point=self.elements_all["translation_large"],
                    base_point=self.elements_all["translation_small"],
                    expected=self.elements_all["translation_large"]
                    - self.elements_all["translation_small"],
                ),
            ]
            return self.generate_tests(smoke_data)

        def regularize_extreme_cases_test_data(self):
            smoke_data = []
            for angle_type in ["angle_close_0", "angle_close_pi_low", "angle_0"]:
                point = self.elements_all[angle_type]
                smoke_data += [dict(point=point, expected=point)]

            if not geomstats.tests.tf_backend():
                angle_type = "angle_pi"
                point = self.elements_all[angle_type]
                smoke_data += [dict(point=point, expected=point)]

                angle_type = "angle_close_pi_high"
                point = self.elements_all[angle_type]

                norm = gs.linalg.norm(point[:3])
                expected_rot = gs.concatenate(
                    [point[:3] / norm * (norm - 2 * gs.pi), gs.zeros(3)], axis=0
                )
                expected_trans = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
                expected = expected_rot + expected_trans
                smoke_data += [dict(point=point, expected=expected)]

                in_pi_2pi = ["angle_in_pi_2pi", "angle_close_2pi_low"]

                for angle_type in in_pi_2pi:
                    point = self.elements_all[angle_type]
                    angle = gs.linalg.norm(point[:3])
                    new_angle = gs.pi - (angle - gs.pi)

                    expected_rot = gs.concatenate(
                        [-new_angle * (point[:3] / angle), gs.zeros(3)], axis=0
                    )
                    expected_trans = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
                    expected = expected_rot + expected_trans
                    smoke_data += [dict(point=point, expected=expected)]

                angle_type = "angle_2pi"
                point = self.elements_all[angle_type]

                expected = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
                smoke_data += [dict(point=point, expected=expected)]

                angle_type = "angle_close_2pi_high"
                point = self.elements_all[angle_type]
                angle = gs.linalg.norm(point[:3])
                new_angle = angle - 2 * gs.pi

                expected_rot = gs.concatenate(
                    [new_angle * point[:3] / angle, gs.zeros(3)], axis=0
                )
                expected_trans = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
                expected = expected_rot + expected_trans
                smoke_data += [dict(point=point, expected=expected)]
            return self.generate_tests(smoke_data)

    testing_data = SpecialEuclidean3VectorsTestData()

    @geomstats.tests.np_and_autograd_only
    def test_log_then_exp(self, metric, point, base_point):
        """
        Test that the Riemannian right exponential and the
        Riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        group = SpecialEuclidean(3, "vector")
        result = metric.exp(metric.log(point, base_point), base_point)
        expected = group.regularize(point)
        expected = gs.cast(expected, gs.float64)
        norm = gs.linalg.norm(expected)
        atol = ATOL
        if norm != 0:
            atol = ATOL * norm
        self.assertAllClose(result, expected, atol=atol)

    @geomstats.tests.np_and_autograd_only
    def test_log_then_exp_right_with_angles_close_to_pi(
        self, metric, point, base_point
    ):
        group = SpecialEuclidean(3, "vector")
        result = metric.exp(metric.log(point, base_point), base_point)
        expected = group.regularize(point)

        inv_expected = gs.concatenate([-expected[:3], expected[3:6]])

        norm = gs.linalg.norm(expected)
        atol = ATOL
        if norm != 0:
            atol = ATOL * norm

        self.assertTrue(
            gs.allclose(result, expected, atol=atol)
            or gs.allclose(result, inv_expected, atol=atol)
        )

    @geomstats.tests.np_and_autograd_only
    def test_exp_then_log_with_angles_close_to_pi(
        self, metric, tangent_vec, base_point
    ):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        group = SpecialEuclidean(3, "vector")
        result = metric.log(metric.exp(tangent_vec, base_point), base_point)

        expected = group.regularize_tangent_vec(
            tangent_vec=tangent_vec, base_point=base_point, metric=metric
        )

        inv_expected = gs.concatenate([-expected[:3], expected[3:6]])

        norm = gs.linalg.norm(expected)
        atol = ATOL
        if norm != 0:
            atol = ATOL * norm

        self.assertTrue(
            gs.allclose(result, expected, atol=atol)
            or gs.allclose(result, inv_expected, atol=atol)
        )

    @geomstats.tests.np_and_autograd_only
    def test_exp_then_log(self, metric, tangent_vec, base_point):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        group = SpecialEuclidean(3, "vector")
        result = metric.log(metric.exp(tangent_vec, base_point), base_point)

        expected = group.regularize_tangent_vec(
            tangent_vec=tangent_vec, base_point=base_point, metric=metric
        )

        norm = gs.linalg.norm(expected)
        atol = ATOL
        if norm != 0:
            atol = ATOL * norm
        self.assertAllClose(result, expected, atol=atol)

    @geomstats.tests.np_and_autograd_only
    def test_exp(self, metric, base_point, tangent_vec, expected):
        result = metric.exp(base_point=base_point, tangent_vec=tangent_vec)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_log(self, metric, point, base_point, expected):
        result = metric.log(point, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_regularize_extreme_cases(self, point, expected):
        group = SpecialEuclidean(3, "vector")
        result = group.regularize(point)
        self.assertAllClose(result, expected)
