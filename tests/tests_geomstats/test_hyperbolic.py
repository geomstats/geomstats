"""Unit tests for the Hyperbolic space."""

import math

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.poincare_ball import PoincareBall

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array

RTOL = 1e-6


class TestHyperbolic(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 3
        self.space = Hyperboloid(dim=self.dimension)
        self.metric = self.space.metric
        self.ball_manifold = PoincareBall(dim=2)
        self.n_samples = 10

    def test_belongs_intrinsic(self):
        self.space.coords_type = "intrinsic"
        point = gs.random.rand(self.n_samples, self.dimension)
        result = self.space.belongs(point)
        self.assertTrue(gs.all(result))

    def test_regularize_intrinsic(self):
        self.space.coords_type = "intrinsic"
        point = gs.random.rand(self.n_samples, self.dimension)
        regularized = self.space.regularize(point)
        self.space.coords_type = "extrinsic"
        result = self.space.belongs(regularized)
        self.assertTrue(gs.all(result))

    def test_regularize_zero_norm(self):
        point = gs.array([-1.0, 1.0, 0.0, 0.0])
        self.assertRaises(ValueError, lambda: self.space.regularize(point))
        self.assertRaises(
            NameError, lambda: self.space.extrinsic_to_intrinsic_coords(point)
        )

    def test_random_uniform_and_belongs(self):
        point = self.space.random_point()
        result = self.space.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

    def test_random_uniform(self):
        result = self.space.random_point()

        self.assertAllClose(gs.shape(result), (self.dimension + 1,))

    def test_projection_to_tangent_space(self):
        base_point = gs.array([1.0, 0.0, 0.0, 0.0])
        belongs = self.space.belongs(base_point)
        self.assertTrue(belongs)

        tangent_vec = self.space.to_tangent(
            vector=gs.array([1.0, 2.0, 1.0, 3.0]), base_point=base_point
        )
        result = self.metric.inner_product(tangent_vec, base_point)
        expected = 0.0

        self.assertAllClose(result, expected)

        result = self.space.to_tangent(
            vector=gs.array([1.0, 2.0, 1.0, 3.0]), base_point=base_point
        )
        expected = tangent_vec

        self.assertAllClose(result, expected)

    def test_intrinsic_and_extrinsic_coords(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = gs.ones(self.dimension)
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int
        self.assertAllClose(result, expected)

        point_ext = gs.array([2.0, 1.0, 1.0, 1.0])
        point_int = self.space.to_coordinates(point_ext, "intrinsic")
        result = self.space.from_coordinates(point_int, "intrinsic")
        expected = point_ext

        self.assertAllClose(result, expected)

    def test_intrinsic_and_extrinsic_coords_vectorization(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = gs.array(
            [
                [0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
                [0.1, 0.1, 0.1, 0.4, 0.1, 0.0],
                [0.1, 0.3, 0.0, 0.1, 0.0, 0.0],
                [-0.1, 0.1, -0.4, 0.1, -0.01, 0.0],
                [0.0, 0.0, 0.1, 0.1, -0.08, -0.1],
                [0.1, 0.1, 0.1, 0.1, 0.0, -0.5],
            ]
        )
        point_ext = self.space.from_coordinates(point_int, "intrinsic")
        result = self.space.to_coordinates(point_ext, "intrinsic")
        expected = point_int
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

        point_ext = gs.array(
            [
                [2.0, 1.0, 1.0, 1.0],
                [4.0, 1.0, 3.0, math.sqrt(5.0)],
                [3.0, 2.0, 0.0, 2.0],
            ]
        )
        point_int = self.space.to_coordinates(point_ext, "intrinsic")
        result = self.space.from_coordinates(point_int, "intrinsic")
        expected = point_ext
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

    def test_log_and_exp_general_case(self):
        """
        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point = gs.array([4.0, 1.0, 3.0, math.sqrt(5.0)])
        point = gs.array([2.0, 1.0, 1.0, 1.0])

        log = self.metric.log(point=point, base_point=base_point)

        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point
        self.assertAllClose(result, expected)

    def test_log_and_exp_general_case_general_dim(self):
        """
        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        dim = 5
        n_samples = self.n_samples

        h5 = Hyperboloid(dim=dim)
        h5_metric = h5.metric

        base_point = h5.random_point()
        point = h5.random_point()
        point = gs.cast(point, gs.float64)
        base_point = gs.cast(base_point, gs.float64)
        one_log = h5_metric.log(point=point, base_point=base_point)

        result = h5_metric.exp(tangent_vec=one_log, base_point=base_point)
        expected = point
        self.assertAllClose(result, expected)

        # Test vectorization of log
        base_point = gs.stack([base_point] * n_samples, axis=0)
        point = gs.stack([point] * n_samples, axis=0)
        expected = gs.stack([one_log] * n_samples, axis=0)

        log = h5_metric.log(point=point, base_point=base_point)
        result = log

        self.assertAllClose(gs.shape(result), (n_samples, dim + 1))
        self.assertAllClose(result, expected)

        result = h5_metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(gs.shape(result), (n_samples, dim + 1))
        self.assertAllClose(result, expected)

        # Test vectorization of exp
        tangent_vec = gs.stack([one_log] * n_samples, axis=0)
        exp = h5_metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = exp

        expected = point
        self.assertAllClose(gs.shape(result), (n_samples, dim + 1))
        self.assertAllClose(result, expected)

    def test_exp_and_belongs(self):
        H2 = Hyperboloid(dim=2)
        METRIC = H2.metric

        base_point = gs.array([1.0, 0.0, 0.0])
        self.assertTrue(H2.belongs(base_point))

        tangent_vec = H2.to_tangent(
            vector=gs.array([1.0, 2.0, 1.0]), base_point=base_point
        )
        exp = METRIC.exp(tangent_vec=tangent_vec, base_point=base_point)
        self.assertTrue(H2.belongs(exp))

    def test_exp_small_vec(self):
        H2 = Hyperboloid(dim=2)
        METRIC = H2.metric

        base_point = H2.regularize(gs.array([1.0, 0.0, 0.0]))
        self.assertTrue(H2.belongs(base_point))

        tangent_vec = 1e-9 * H2.to_tangent(
            vector=gs.array([1.0, 2.0, 1.0]), base_point=base_point
        )
        exp = METRIC.exp(tangent_vec=tangent_vec, base_point=base_point)
        self.assertTrue(H2.belongs(exp))

    def test_exp_vectorization(self):
        n_samples = 3
        dim = self.dimension + 1

        one_vec = gs.array([2.0, 1.0, 1.0, 1.0])
        one_base_point = gs.array([4.0, 3.0, 1.0, math.sqrt(5)])
        n_vecs = gs.array(
            [
                [2.0, 1.0, 1.0, 1.0],
                [4.0, 1.0, 3.0, math.sqrt(5.0)],
                [3.0, 2.0, 0.0, 2.0],
            ]
        )
        n_base_points = gs.array(
            [
                [2.0, 0.0, 1.0, math.sqrt(2)],
                [5.0, math.sqrt(8), math.sqrt(8), math.sqrt(8)],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        one_tangent_vec = self.space.to_tangent(one_vec, base_point=one_base_point)
        result = self.metric.exp(one_tangent_vec, one_base_point)
        self.assertAllClose(gs.shape(result), (dim,))

        n_tangent_vecs = self.space.to_tangent(n_vecs, base_point=one_base_point)
        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        expected = []

        for i in range(n_samples):
            expected.append(self.metric.exp(n_tangent_vecs[i], one_base_point))
        expected = gs.stack(expected, axis=0)
        expected = helper.to_vector(gs.array(expected))
        self.assertAllClose(result, expected, atol=1e-2)

        one_tangent_vec = self.space.to_tangent(one_vec, base_point=n_base_points)
        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        expected = []
        for i in range(n_samples):
            expected.append(self.metric.exp(one_tangent_vec[i], n_base_points[i]))
        expected = gs.stack(expected, axis=0)
        expected = helper.to_vector(gs.array(expected))
        self.assertAllClose(result, expected)

        n_tangent_vecs = self.space.to_tangent(n_vecs, base_point=n_base_points)
        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        expected = []
        for i in range(n_samples):
            expected.append(self.metric.exp(n_tangent_vecs[i], n_base_points[i]))
        expected = gs.stack(expected, axis=0)
        expected = helper.to_vector(gs.array(expected))
        self.assertAllClose(result, expected)

    def test_log_vectorization(self):
        n_samples = 3
        dim = self.dimension + 1

        one_point = gs.array([2.0, 1.0, 1.0, 1.0])
        one_base_point = gs.array([4.0, 3.0, 1.0, math.sqrt(5)])
        n_points = gs.array(
            [[2.0, 1.0, 1.0, 1.0], [4.0, 1.0, 3.0, math.sqrt(5)], [3.0, 2.0, 0.0, 2.0]]
        )
        n_base_points = gs.array(
            [
                [2.0, 0.0, 1.0, math.sqrt(2)],
                [5.0, math.sqrt(8), math.sqrt(8), math.sqrt(8)],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        result = self.metric.log(one_point, one_base_point)
        self.assertAllClose(gs.shape(result), (dim,))

        result = self.metric.log(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

    def test_inner_product(self):
        """
        Test that the inner product between two tangent vectors
        is the Minkowski inner product.
        """
        minkowski_space = Minkowski(self.dimension + 1)
        base_point = gs.array([1.16563816, 0.36381045, -0.47000603, 0.07381469])

        tangent_vec_a = self.space.to_tangent(
            vector=gs.array([10.0, 200.0, 1.0, 1.0]), base_point=base_point
        )

        tangent_vec_b = self.space.to_tangent(
            vector=gs.array([11.0, 20.0, -21.0, 0.0]), base_point=base_point
        )

        result = self.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)

        expected = minkowski_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

        self.assertAllClose(result, expected)

    def test_squared_norm_and_squared_dist(self):
        """
        Test that the squared distance between two points is
        the squared norm of their logarithm.
        """
        point_a = gs.array([2.0, 1.0, 1.0, 1.0])
        point_b = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.squared_norm(vector=log)
        expected = self.metric.squared_dist(point_a, point_b)

        self.assertAllClose(result, expected)

    def test_norm_and_dist(self):
        """
        Test that the distance between two points is
        the norm of their logarithm.
        """
        point_a = gs.array([2.0, 1.0, 1.0, 1.0])
        point_b = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.norm(vector=log)
        expected = self.metric.dist(point_a, point_b)

        self.assertAllClose(result, expected)

    def test_log_and_exp_edge_case(self):
        """
        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point_intrinsic = gs.array([1.0, 2.0, 3.0])
        base_point = self.space.from_coordinates(base_point_intrinsic, "intrinsic")
        point_intrinsic = base_point_intrinsic + 1e-12 * gs.array([-1.0, -2.0, 1.0])
        point = self.space.from_coordinates(point_intrinsic, "intrinsic")

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
        """
        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # General case
        base_point = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
        vector = gs.array([2.0, 1.0, 1.0, 1.0])
        vector = self.space.to_tangent(vector=vector, base_point=base_point)
        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)

        expected = vector
        self.assertAllClose(result, expected)

    def test_dist(self):
        # Distance between a point and itself is 0.
        point_a = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
        point_b = point_a
        result = self.metric.dist(point_a, point_b)
        expected = 0
        self.assertAllClose(result, expected)

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        base_point = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
        vector = gs.array([0.001, 0.0, -0.00001, -0.00003])
        tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)
        exp = self.metric.exp(tangent_vec=tangent_vec, base_point=base_point)

        result = self.metric.dist(base_point, exp)
        sq_norm = self.metric.embedding_metric.squared_norm(tangent_vec)
        expected = sq_norm
        self.assertAllClose(result, expected, atol=1e-2)

    def test_geodesic_and_belongs(self):
        initial_point = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
        n_geodesic_points = 100
        vector = gs.array([1.0, 0.0, 0.0, 0.0])

        initial_tangent_vec = self.space.to_tangent(
            vector=vector, base_point=initial_point
        )
        geodesic = self.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        t = gs.linspace(start=0.0, stop=1.0, num=n_geodesic_points)
        points = geodesic(t)
        result = gs.all(self.space.belongs(points))
        self.assertTrue(result)

    def test_geodesic_and_belongs_large_initial_velocity(self):
        initial_point = gs.array([4.0, 1.0, 3.0, math.sqrt(5)])
        n_geodesic_points = 100
        vector = gs.array([2.0, 0.0, 0.0, 0.0])

        initial_tangent_vec = self.space.to_tangent(
            vector=vector, base_point=initial_point
        )
        geodesic = self.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        t = gs.linspace(start=0.0, stop=1.0, num=n_geodesic_points)
        points = geodesic(t)
        result = gs.all(self.space.belongs(points, atol=gs.atol * 1e4))
        self.assertTrue(result)

    def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
        """
        Test that the Riemannian exponential and
        the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        base_point = gs.array([2.0, 1.0, 1.0, 1.0])
        vector = 1e-10 * gs.array([0.06, -51.0, 6.0, 5.0])

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)
        expected = self.space.to_tangent(vector=vector, base_point=base_point)

        self.assertAllClose(result, expected)

    def test_scaled_inner_product(self):
        base_point_intrinsic = gs.array([1.0, 1.0, 1.0])
        base_point = self.space.from_coordinates(base_point_intrinsic, "intrinsic")
        tangent_vec_a = gs.array([1.0, 2.0, 3.0, 4.0])
        tangent_vec_b = gs.array([5.0, 6.0, 7.0, 8.0])
        tangent_vec_a = self.space.to_tangent(tangent_vec_a, base_point)
        tangent_vec_b = self.space.to_tangent(tangent_vec_b, base_point)
        scale = 2
        default_space = Hyperboloid(dim=self.dimension)
        scaled_space = Hyperboloid(dim=self.dimension, scale=2)
        inner_product_default_metric = default_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        inner_product_scaled_metric = scaled_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        result = inner_product_scaled_metric
        expected = scale ** 2 * inner_product_default_metric
        self.assertAllClose(result, expected)

    def test_scaled_squared_norm(self):
        base_point_intrinsic = gs.array([1.0, 1.0, 1.0])
        base_point = self.space.from_coordinates(base_point_intrinsic, "intrinsic")
        tangent_vec = gs.array([1.0, 2.0, 3.0, 4.0])
        tangent_vec = self.space.to_tangent(tangent_vec, base_point)
        scale = 2
        default_space = Hyperboloid(dim=self.dimension)
        scaled_space = Hyperboloid(dim=self.dimension, scale=2)
        squared_norm_default_metric = default_space.metric.squared_norm(
            tangent_vec, base_point
        )
        squared_norm_scaled_metric = scaled_space.metric.squared_norm(
            tangent_vec, base_point
        )
        result = squared_norm_scaled_metric
        expected = scale ** 2 * squared_norm_default_metric
        self.assertAllClose(result, expected)

    def test_scaled_distance(self):
        point_a_intrinsic = gs.array([1.0, 2.0, 3.0])
        point_b_intrinsic = gs.array([4.0, 5.0, 6.0])
        point_a = self.space.from_coordinates(point_a_intrinsic, "intrinsic")
        point_b = self.space.from_coordinates(point_b_intrinsic, "intrinsic")
        scale = 2
        scaled_space = Hyperboloid(dim=self.dimension, scale=2)
        distance_default_metric = self.space.metric.dist(point_a, point_b)
        distance_scaled_metric = scaled_space.metric.dist(point_a, point_b)
        result = distance_scaled_metric
        expected = scale * distance_default_metric
        self.assertAllClose(result, expected)

    def test_is_tangent(self):
        base_point = gs.array([4.0, 1.0, 3.0, math.sqrt(5.0)])
        point = gs.array([2.0, 1.0, 1.0, 1.0])

        log = self.metric.log(point=point, base_point=base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

    @geomstats.tests.np_autograd_and_tf_only
    def test_parallel_transport_vectorization(self):
        space = self.space
        shape = (4, space.dim + 1)
        metric = space.metric

        results = helper.test_parallel_transport(space, metric, shape)
        for res in results:
            self.assertTrue(res)

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.dimension + 1)
        result = helper.test_projection_and_belongs(
            self.space, shape, atol=gs.atol * 100
        )
        for res in result:
            self.assertTrue(res)

        point = gs.array([0.0, 1.0, 0.0, 0.0])
        projected = self.space.projection(point)
        result = self.space.belongs(projected)
        self.assertTrue(result)
