"""Unit tests for ProductManifold."""
import random

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import (
    NFoldManifold,
    NFoldMetric,
    ProductManifold,
)
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import TestCase
from tests.data_generation import (
    RiemannianMetricTestData,
    TestData,
    VectorSpaceTestData,
)
from tests.parametrizers import (
    ManifoldParametrizer,
    RiemannianMetricParametrizer,
    VectorSpaceParametrizer,
)

smoke_manifolds_1 = [Hypersphere(dim=2), Hyperboloid(dim=2)]


# class TestProductManifold(TestCase, metaclass=ManifoldParametrizer):
#     space = ProductManifold

#     class TestDataProductManifold(TestData):
#         def dimension_data(self):
#             smoke_data = [
#                 dict(
#                     manifold=smoke_manifolds_1,
#                     default_point_type="vector",
#                     expected=4,
#                 ),
#                 dict(
#                     manifold=smoke_manifolds_1,
#                     default_point_type="matrix",
#                     expected=4,
#                 ),
#             ]
#             return self.generate_tests(smoke_data)

#         def regularize_data(self):
#             smoke_data = [
#                 dict(
#                     manifold=smoke_manifolds_1,
#                     default_point_type="vector",
#                     point=ProductManifold(
#                         self.smoke_manifolds, default_point_type="vector"
#                     ).random_point(5),
#                 ),
#                 dict(
#                     manifold=smoke_manifolds_1,
#                     default_point_type="matrix",
#                     point=ProductManifold(
#                         self.smoke_manifolds, default_point_type="matrix"
#                     ).random_point(5),
#                 ),
#             ]

#     def test_dimension(self, manifolds, default_point_type, expected):
#         space = self.space(manifolds, default_point_type=default_point_type)
#         self.assertAllClose(space.dim, expected)

#     def test_regularize(self, manifolds, default_point_type, point):
#         space = self.space(manifolds, default_point_type=default_point_type)
#         result = self.space_vector.regularize(point)
#         self.assertAllClose(result, point)

#     @geomstats.tests.np_autograd_and_torch_only
#     def test_inner_product_matrix_matrix(self):
#         euclidean = Euclidean(3)
#         minkowski = Minkowski(3)
#         space = ProductManifold(
#             manifolds=[euclidean, minkowski], default_point_type="matrix"
#         )
#         point = space.random_point(1)
#         result = space.metric.metric_matrix(point)
#         expected = gs.eye(6)
#         expected[3, 3] = -1
#         self.assertAllClose(result, expected)

#     @geomstats.tests.np_autograd_and_torch_only
#     def test_inner_product_matrix_vector(self):
#         euclidean = Euclidean(3)
#         minkowski = Minkowski(3)
#         space = ProductManifold(
#             manifolds=[euclidean, minkowski], default_point_type="vector"
#         )
#         point = space.random_point(1)
#         expected = gs.eye(6)
#         expected[3, 3] = -1
#         result = space.metric.metric_matrix(point)
#         self.assertAllClose(result, expected)


# class TestProductRiemannianMetric(TestCase, metaclass=RiemannianMetricParametrizer):
#     class TestDataProductRiemannianMetric(TestData):
#         def dist_log_exp_norm_vector_data(self):
#             smoke_data = [
#                 dict(
#                     manifold=smoke_manifolds_1,
#                     default_point_type="vector",
#                     point=ProductManifold(
#                         smoke_manifolds_1, default_point_type="vector"
#                     ).random_point(5),
#                     base_point=ProductManifold(
#                         smoke_manifolds_1, default_point_type="vector"
#                     ).random_point(),
#                 ),
#                 dict(
#                     manifold=smoke_manifolds_1,
#                     default_point_type="matrix",
#                     point=ProductManifold(
#                         smoke_manifolds_1, default_point_type="matrix"
#                     ).random_point(5),
#                     base_point=ProductManifold(
#                         smoke_manifolds_1, default_point_type="matrix"
#                     ).random_point(),
#                 ),
#             ]
#             return self.generate_tests(smoke_data)

#         def inner_product_matrix_data(self):
#             return self.dist_log_exp_norm_vector_data()

#     testing_data = TestDataProductRiemannianMetric()

#     def test_dist_log_exp_norm_vector(
#         self, manifolds, default_point_type, point, base_point
#     ):
#         metric = self.metric(manifolds, default_point_type=default_point_type)

#         logs = metric.log(point, base_point)
#         normalized_logs = gs.einsum(
#             "..., ...j->...j",
#             1.0 / metric.norm(logs, base_point),
#             logs,
#         )
#         point = metric.exp(normalized_logs, base_point)
#         result = metric.dist(point, base_point)

#         expected = gs.ones(len(result))
#         self.assertAllClose(result, expected)

#     def test_inner_product_matrix(
#         self, manifolds, default_point_type, point, base_point
#     ):
#         metric = self.metric(manifolds, default_point_type=default_point_type)
#         logs = metric.log(point, base_point)
#         result = metric.inner_product(logs, logs)
#         expected = metric.squared_dist(base_point, point)
#         self.assertAllClose(result, expected)


# class TestNFoldManifold(TestCase, metaclass=ManifoldParametrizer):
#     space = NFoldManifold

#     class TestDataNFoldManifold:
#         def belongs_data(self):
#             smoke_data = [
#                 dict(
#                     base=SpecialOrthogonal(3),
#                     power=2,
#                     point=gs.stack([gs.eye(3) + 1.0, gs.eye(3)]),
#                     expected=False,
#                 ),
#                 dict(
#                     base=SpecialOrthogonal(3),
#                     power=2,
#                     point=gs.array([gs.eye(3)]*2),
#                     expected=[True, True],
#                 )
#             ]
#             return self.generate_tests(smoke_data)

#         def shape_data(self):
#             smoke_data =  [
#                 dict(
#                     base=SpecialOrthogonal(3),
#                     power=2,
#                     shape=(2, 3, 3)
#                 )
#             ]
#             return self.generate_tests(smoke_data)

#     def test_belongs(self, base, power, point, expected):
#         space = self.space(base, power)
#         self.assertAllClose(space.belongs(point), expected)

#     def test_shape(self, base, power, expected):
#         space = self.space(base, power)
#         self.assertAllClose(space.shape, expected)


#     testing_data = TestDataNFoldManifold()


class TestNFoldMetric(TestCase, metaclass=RiemannianMetricParametrizer):
    metric = connection = NFoldMetric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_log_is_tangent = True
    skip_test_squared_dist_is_symmetric = True

    class TestDataNFoldMetric(RiemannianMetricTestData):

        n_list = random.sample(range(3, 5), 2)
        power_list = random.sample(range(2, 5), 2)
        base_list = [SpecialOrthogonal(n) for n in n_list]
        metric_args_list = list(zip(base_list, power_list))
        shape_list = [(power, n, n) for n, power in zip(n_list, power_list)]
        space_list = [
            NFoldManifold(base, power) for base, power in zip(base_list, power_list)
        ]
        n_points_list = random.sample(range(1, 5), 2)
        n_samples_list = random.sample(range(1, 5), 2)
        n_points_a_list = random.sample(range(1, 5), 2)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

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
                is_tangent_atol=1e-1,
            )

        def geodesic_ivp_belongs_data(self):
            return self._geodesic_ivp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_data(self):
            return self._geodesic_bvp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
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
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
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

    testing_data = TestDataNFoldMetric()
