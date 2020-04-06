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

    @geomstats.tests.np_and_pytorch_only
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
