"""Unit tests for the Hyperbolic space using Poincaré Ball Model.

We verify poincare ball model by compare results
of squared distance computed with inner_product
(using RiemannianMetric methods) and distance defined
in PoincareBall.
We also verify the distance is the same using differents
coordinates systems.

"""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall


class TestPoincareBallMethods(geomstats.tests.TestCase):
    def setUp(self):
        self.manifold = PoincareBall(2)
        self.metric = self.manifold.metric

        self.hyperboloid_manifold = Hyperboloid(2)
        self.hyperboloid_metric = self.hyperboloid_manifold.metric

    @geomstats.tests.np_only
    def test_squared_dist(self):
        point_a = gs.array([[-0.3, 0.7]])
        point_b = gs.array([[0.2, 0.5]])

        distance_a_b = self.metric.dist(point_a, point_b)
        squared_distance = self.metric.squared_dist(point_a, point_b)

        self.assertAllClose(distance_a_b**2, squared_distance, atol=1e-8)

    @geomstats.tests.np_and_pytorch_only
    def test_coordinates(self):
        point_a = gs.array([[-0.3, 0.7]])
        point_b = gs.array([[0.2, 0.5]])

        point_a_h =\
            self.manifold.to_coordinates(point_a, 'extrinsic')
        point_b_h =\
            self.manifold.to_coordinates(point_b, 'extrinsic')

        dist_in_ball =\
            self.metric.dist(point_a, point_b)
        dist_in_hype =\
            self.hyperboloid_metric.dist(point_a_h, point_b_h)

        self.assertAllClose(dist_in_ball, dist_in_hype, atol=1e-8)

    @geomstats.tests.np_and_pytorch_only
    def test_dist_poincare(self):

        point_a = gs.array([0.5, 0.5])
        point_b = gs.array([0.5, -0.5])

        dist_a_b =\
            self.manifold.metric.dist(point_a, point_b)

        result = dist_a_b
        expected = gs.array([[2.887270927429199]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_dist_vectorization(self):
        point_a = gs.array([0.2, 0.5])
        point_b = gs.array([[0.3, -0.5], [0.2, 0.2]])

        dist_a_b =\
            self.manifold.metric.dist(point_a, point_b)

        result_vect = dist_a_b
        result =\
            [self.manifold.metric.dist(point_a, point_b[i])
             for i in range(len(point_b))]
        result = gs.concatenate(result, axis=0)
        self.assertAllClose(result_vect, result)

    def test_mobius_vectorization(self):
        point_a = gs.array([0.5, 0.5])
        point_b = gs.array([[0.5, -0.3], [0.3, 0.4]])

        dist_a_b =\
            self.manifold.metric.mobius_add(point_a, point_b)

        result_vect = dist_a_b
        result =\
            [self.manifold.metric.mobius_add(point_a, point_b[i])
             for i in range(len(point_b))]
        result = gs.concatenate(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b =\
            self.manifold.metric.mobius_add(point_b, point_a)

        result_vect = dist_a_b
        result =\
            [self.manifold.metric.mobius_add(point_b[i], point_a)
             for i in range(len(point_b))]
        result = gs.concatenate(result, axis=0)
        self.assertAllClose(result_vect, result)

    @geomstats.tests.np_only
    def test_log_vectorization(self):
        point_a = gs.array([0.5, 0.5])
        point_b = gs.array([[0.5, -0.5], [0.4, 0.4]])

        dist_a_b =\
            self.manifold.metric.log(point_a, point_b)

        result_vect = dist_a_b
        result =\
            [self.manifold.metric.log(point_a, point_b[i])
             for i in range(len(point_b))]
        result = gs.concatenate(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b =\
            self.manifold.metric.log(point_b, point_a)

        result_vect = dist_a_b
        result =\
            [self.manifold.metric.log(point_b[i], point_a)
             for i in range(len(point_b))]
        result = gs.concatenate(result, axis=0)
        self.assertAllClose(result_vect, result)

    @geomstats.tests.np_and_pytorch_only
    def test_exp_vectorization(self):
        point_a = gs.array([0.5, 0.5])
        point_b = gs.array([[0.5, -0.5], [0.4, 0.4]])

        dist_a_b =\
            self.manifold.metric.exp(point_a, point_b)

        result_vect = dist_a_b
        result =\
            [self.manifold.metric.exp(point_a, point_b[i])
             for i in range(len(point_b))]
        result = gs.concatenate(result, axis=0)
        self.assertAllClose(result_vect, result)

        dist_a_b =\
            self.manifold.metric.exp(point_b, point_a)

        result_vect = dist_a_b
        result =\
            [self.manifold.metric.exp(point_b[i], point_a)
             for i in range(len(point_b))]
        result = gs.concatenate(result, axis=0)
        self.assertAllClose(result_vect, result)

    @geomstats.tests.np_only
    def test_log_poincare(self):

        point = gs.array([[0.3, 0.5]])
        base_point = gs.array([[0.3, 0.3]])

        result = self.manifold.metric.log(point, base_point)
        expected = gs.array([[-0.01733576, 0.21958634]])

        self.manifold.metric.coords_type = 'extrinsic'
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_belong_true_poincare(self):
        point = gs.array([[0.3, 0.5]])
        belong = self.manifold.belongs(point)
        self.assertTrue(belong)

    @geomstats.tests.np_only
    def test_belong_false_poincare(self):
        point = gs.array([[1.2, 0.5]])
        belong = self.manifold.belongs(point)
        self.assertFalse(belong)

    @geomstats.tests.np_only
    def test_exp_poincare(self):

        point = gs.array([[0.3, 0.5]])
        base_point = gs.array([[0.3, 0.3]])

        tangent_vec = self.manifold.metric.log(point, base_point)
        result = self.manifold.metric.exp(tangent_vec, base_point)

        self.manifold.metric.coords_type = 'extrinsic'
        self.assertAllClose(result, point)

    @geomstats.tests.np_and_pytorch_only
    def test_ball_retraction(self):
        x = gs.array([[0.5, 0.6], [0.2, -0.1], [0.2, -0.4]])
        y = gs.array([[0.3, 0.5], [0.3, -0.6], [0.3, -0.3]])

        ball_metric = self.manifold.metric
        tangent_vec = ball_metric.log(y, x)
        ball_metric.retraction(tangent_vec, x)
