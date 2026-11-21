import geomstats.backend as gs
from geomstats.metric_geometry.wald_space import (
    _squared_dist_and_grad_autodiff,
    make_topologies,
)
from geomstats.test.test_case import TestCase
from geomstats.test_cases.metric_geometry.point_set import PointTestCase


class MakeTopologiesTestCase(TestCase):
    def test_number_of_topologies(self, n_labels, expected):
        """Check number of topologies.

        Parameters
        ----------
        n_labels : int
            Number of labels.
        expected : int
            Expected number of topologies.
        """
        topologies = list(make_topologies(n_labels))
        self.assertTrue(len(topologies) == expected)


class SquaredDistAndGradTestCase(TestCase):
    def test_value_and_grad_against_autodiff(
        self, squared_dist_and_grad, AmbientMetric, atol=gs.atol
    ):
        """Check closed-form squared distance and gradient against autodiff.

        Parameters
        ----------
        squared_dist_and_grad : callable
            Closed-form squared distance and gradient for `AmbientMetric`.
        AmbientMetric : type
            Metric of the ambient space.
        atol : float
            Absolute tolerance.

        Notes
        -----
        The topology, point and weights are sampled at random.
        """
        topology = self.space.random_point().topology

        point = self.space.random_grove_point(topology, n_samples=1)
        ambient_point = self.space.lift(point)

        value_and_grad = squared_dist_and_grad(self.space, topology, ambient_point)
        value_and_grad_autodiff = _squared_dist_and_grad_autodiff(
            self.space, topology, ambient_point
        )

        weights = gs.random.uniform(size=(topology.n_splits,))

        self.space.ambient_space.equip_with_metric(AmbientMetric)
        value, grad = value_and_grad(weights)
        value_autodiff, grad_autodiff = value_and_grad_autodiff(weights)

        self.assertAllClose(value, value_autodiff, atol=atol)
        self.assertAllClose(grad, grad_autodiff, atol=atol)


class WaldTestCase(PointTestCase):
    def test_corr(self, point, expected, atol=gs.atol):
        """Check correlation matrix of a wald.

        Parameters
        ----------
        point : Wald
            Wald.
        expected : array-like, shape=[n_labels, n_labels]
            Expected correlation matrix.
        atol : float
            Absolute tolerance.
        """
        self.assertAllClose(point.corr, expected, atol=atol)


class WaldGeodesicSolverTestCase(TestCase):
    def test_discrete_geodesic_reverse(self, initial_point, end_point, atol=gs.atol):
        """Check discrete geodesic is the reverse of the reversed one.

        Parameters
        ----------
        initial_point : Wald or WaldBatch
            Initial point of the geodesic.
        end_point : Wald or WaldBatch
            End point of the geodesic.
        atol : float
            Absolute tolerance.
        """
        geod_points = self.geodesic_solver.discrete_geodesic(initial_point, end_point)
        geod_points_reversed = self.geodesic_solver.discrete_geodesic(
            end_point, initial_point
        )

        if not isinstance(initial_point, list):
            geod_points = [geod_points]
            geod_points_reversed = [geod_points_reversed]

        for geod_points_, geod_points_reversed_ in zip(
            geod_points, geod_points_reversed
        ):
            geod_points_reversed_.reverse()
            self.assertTrue(
                gs.all(geod_points_.equal(geod_points_reversed_, atol=atol))
            )
