"""Unit tests for the Hyperbolic space."""
import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.minkowski import Minkowski
from tests.conftest import TestCase
from tests.data_generation import OpenSetTestData, RiemannianMetricTestData
from tests.parametrizers import OpenSetParametrizer, RiemannianMetricParametrizer

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array

RTOL = 1e-6


class TestHyperbolic(TestCase, metaclass=OpenSetParametrizer):
    space = Hyperboloid

    class TestDataHyperbolic(OpenSetTestData):

        smoke_space_args_list = [(2,), (3,), (4,), (5,)]
        smoke_n_points_list = [1, 2, 1, 2]
        n_list = random.sample(range(2, 10), 5)
        space_args_list = [(n,) for n in n_list]
        n_points_list = random.sample(range(1, 10), 5)
        shape_list = [(n,) for n in n_list]
        n_vecs_list = random.sample(range(1, 10), 5)
        n_samples_list = random.sample(range(1, 10), 5)

        def belongs_data(self):
            smoke_data = [
                dict(dim=3, vec=gs.array([1.0, 0.0, 0.0, 0.0]), expected=True)
            ]
            return self.generate_tests(smoke_data)

        def regularize_raises_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    point=gs.array([-1.0, 1.0, 0.0, 0.0]),
                    expected=pytest.raises(ValueError),
                )
            ]
            return self.generate_tests(smoke_data)

        def extrinsic_to_intrinsic_coords_rasises_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    point=gs.array([-1.0, 1.0, 0.0, 0.0]),
                    expected=pytest.raises(ValueError),
                )
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            belongs_atol = gs.atol * 100000
            return self._random_point_belongs_data(
                self.smoke_space_args_list,
                self.smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
                belongs_atol,
            )

        def to_tangent_is_tangent_data(self):

            is_tangent_atol = gs.atol * 1000

            return self._to_tangent_is_tangent_data(
                Hyperboloid,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_in_ambient_space_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_data(
                Hyperboloid, self.space_args_list, self.shape_list
            )

    testing_data = TestDataHyperbolic()

    def test_belongs(self, dim, vec, expected):
        space = self.space(dim)
        self.assertTrue(space.belongs(gs.array(vec)), gs.array(expected))

    def test_regularize_raises(self, dim, point, expected):
        space = self.space(dim)
        with expected:
            space.regularize(point)

    def test_extrinsic_to_intrinsic_coords_rasises(self, dim, point, expected):
        space = self.space(dim)
        with expected:
            space.extrinsic_to_intrinsic_coords(point)


# class TestHyperboloidMetric(TestCase, metaclass=RiemannianMetricParametrizer):
#     class TestDataHyperboloidMetric(RiemannianMetricTestData):
#         def inner_product_is_minkowski_inner_product_data(self):
#             base_point = gs.array([1.16563816, 0.36381045, -0.47000603, 0.07381469])
#             tangent_vec_a = self.space.to_tangent(
#                 vector=gs.array([10.0, 200.0, 1.0, 1.0]), base_point=base_point
#             )
#             tangent_vec_b = self.space.to_tangent(
#                 vector=gs.array([11.0, 20.0, -21.0, 0.0]), base_point=base_point
#             )
#             smoke_data = [
#                 dict(
#                     dim=3,
#                     tangent_vec_a=tangent_vec_a,
#                     tangent_vec_b=tangent_vec_b,
#                     base_point=base_point,
#                 )
#             ]
#             return self.generate_tests(smoke_data)

#         def scaled_inner_product_data(self):
#             space = Hyperboloid(3)
#             base_point = space.from_coordinates(gs.array([1.0, 1.0, 1.0]), "intrinsic")
#             tangent_vec_a = space.to_tangent(gs.array([1.0, 2.0, 3.0, 4.0]), base_point)
#             tangent_vec_b = space.to_tangent(gs.array([5.0, 6.0, 7.0, 8.0]), base_point)
#             smoke_data = [
#                 dict(
#                     dim=3,
#                     scale=2,
#                     tangent_vec_a=tangent_vec_a,
#                     tangent_vec_b=tangent_vec_b,
#                     base_point=base_point,
#                 )
#             ]
#             return self.generate_tests(smoke_data)

#         def scaled_squared_norm_data(self):
#             space = Hyperboloid(3)
#             base_point = space.from_coordinates(gs.array([1.0, 1.0, 1.0]), "intrinsic")
#             tangent_vec = space.to_tangent(gs.array([1.0, 2.0, 3.0, 4.0]), base_point)
#             smoke_data = [
#                 dict(dim=3, scale=2, tangent_vec=tangent_vec, base_point=base_point)
#             ]
#             return self.generate_tests(smoke_data)

#         def scaled_dist_data(self):
#             space = Hyperboloid(3)
#             point_a = space.from_coordinates(gs.array([1.0, 2.0, 3.0]), "intrinsic")
#             point_b = space.from_coordinates(gs.array([4.0, 5.0, 6.0]), "intrinsic")
#             smoke_data = [dict(dim=3, point_a=point_a, point_b=point_b)]
#             return self.generate_tests(smoke_data)

#     def test_inner_product_is_minkowski_inner_product(
#         self, dim, tangent_vec_a, tangent_vec_b, base_point
#     ):
#         metric = self.metric(dim)
#         minkowki_space = Minkowski(dim + 1)
#         result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
#         expected = minkowki_space.metric.inner_product(
#             tangent_vec_a, tangent_vec_b, base_point
#         )
#         self.assertAllClose(result, expected)

#     def test_scaled_inner_product(
#         self, dim, scale, tangent_vec_a, tangent_vec_b, base_point
#     ):
#         default_space = Hyperboloid(dim=dim)
#         scaled_space = Hyperboloid(dim=dim, scale=scale)
#         inner_product_default_metric = default_space.metric.inner_product(
#             tangent_vec_a, tangent_vec_b, base_point
#         )
#         inner_product_scaled_metric = scaled_space.metric.inner_product(
#             tangent_vec_a, tangent_vec_b, base_point
#         )
#         result = inner_product_scaled_metric
#         expected = scale**2 * inner_product_default_metric
#         self.assertAllClose(result, expected)

#     def test_scaled_squred_norm(self, dim, scale, tangent_vec, base_point):
#         default_space = Hyperboloid(dim=dim)
#         scaled_space = Hyperboloid(dim=dim, scale=scale)
#         squared_norm_default_metric = default_space.metric.squared_norm(
#             tangent_vec, base_point
#         )
#         squared_norm_scaled_metric = scaled_space.metric.squared_norm(
#             tangent_vec, base_point
#         )
#         result = squared_norm_scaled_metric
#         expected = scale**2 * squared_norm_default_metric
#         self.assertAllClose(result, expected)

#     def test_scaled_dist(self, dim, scale, point_a, point_b):
#         default_space = Hyperboloid(dim=dim)
#         scaled_space = Hyperboloid(dim=dim, scale=scale)
#         distance_default_metric = default_space.metric.dist(point_a, point_b)
#         distance_scaled_metric = scaled_space.metric.dist(point_a, point_b)
#         result = distance_scaled_metric
#         expected = scale * distance_default_metric
#         self.assertAllClose(result, expected)


# class TestHyperbolic(geomstats.tests.TestCase):
#     def setup_method(self):
#         gs.random.seed(1234)
#         self.dimension = 3
#         self.space = Hyperboloid(dim=self.dimension)
#         self.metric = self.space.metric
#         self.ball_manifold = PoincareBall(dim=2)
#         self.n_samples = 10


#     def test_regularize_intrinsic(self):
#         self.space.coords_type = "intrinsic"
#         point = gs.random.rand(self.n_samples, self.dimension)
#         regularized = self.space.regularize(point)
#         self.space.coords_type = "extrinsic"
#         result = self.space.belongs(regularized)
#         self.assertTrue(gs.all(result))


#     def test_exp_small_vec(self):
#         H2 = Hyperboloid(dim=2)
#         METRIC = H2.metric

#         base_point = H2.regularize(gs.array([1.0, 0.0, 0.0]))
#         self.assertTrue(H2.belongs(base_point))

#         tangent_vec = 1e-9 * H2.to_tangent(
#             vector=gs.array([1.0, 2.0, 1.0]), base_point=base_point
#         )
#         exp = METRIC.exp(tangent_vec=tangent_vec, base_point=base_point)
#         self.assertTrue(H2.belongs(exp))


#     def test_log_and_exp_edge_case(self):
#         """
#         Test that the Riemannian exponential
#         and the Riemannian logarithm are inverse.

#         Expect their composition to give the identity function.
#         """
#         # Riemannian Log then Riemannian Exp
#         # Edge case: two very close points, base_point_2 and point_2,
#         # form an angle < epsilon
#         base_point_intrinsic = gs.array([1.0, 2.0, 3.0])
#         base_point = self.space.from_coordinates(base_point_intrinsic, "intrinsic")
#         point_intrinsic = base_point_intrinsic + 1e-12 * gs.array([-1.0, -2.0, 1.0])
#         point = self.space.from_coordinates(point_intrinsic, "intrinsic")

#         log = self.metric.log(point=point, base_point=base_point)
#         result = self.metric.exp(tangent_vec=log, base_point=base_point)
#         expected = point

#         self.assertAllClose(result, expected)


#     def test_geodesic_and_belongs_large_initial_velocity(self):
#         initial_point = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
#         n_geodesic_points = 100
#         vector = gs.array([2.0, 0.0, 0.0, 0.0])

#         initial_tangent_vec = self.space.to_tangent(
#             vector=vector, base_point=initial_point
#         )
#         geodesic = self.metric.geodesic(
#             initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
#         )

#         t = gs.linspace(start=0.0, stop=1.0, num=n_geodesic_points)
#         points = geodesic(t)
#         result = gs.all(self.space.belongs(points, atol=gs.atol * 1e4))
#         self.assertTrue(result)

#     def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
#         """
#         Test that the Riemannian exponential and
#         the Riemannian logarithm are inverse.

#         Expect their composition to give the identity function.
#         """
#         # Riemannian Exp then Riemannian Log
#         # Edge case: tangent vector has norm < epsilon
#         base_point = gs.array([2.0, 1.0, 1.0, 1.0])
#         vector = 1e-10 * gs.array([0.06, -51.0, 6.0, 5.0])

#         exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
#         result = self.metric.log(point=exp, base_point=base_point)
#         expected = self.space.to_tangent(vector=vector, base_point=base_point)

#         self.assertAllClose(result, expected)
