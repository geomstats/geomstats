import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class UniformlySampledPathEnergyTestCase(TestCase):
    """Uniformly sampled path energy test case."""

    def test_energy(self, path, expected, atol=gs.atol):
        """Check energy of a path.

        Parameters
        ----------
        path : array-like, shape=[..., n_times, *space.shape]
            Uniformly sampled path.
        expected : array-like, shape=[...,]
            Expected energy.
        atol : float
            Absolute tolerance.
        """
        res = self.path_energy(path)
        self.assertAllClose(res, expected, atol=atol)

    def test_energy_per_time(self, path, expected, atol=gs.atol):
        """Check energy of a path at each time step.

        Parameters
        ----------
        path : array-like, shape=[..., n_times, *space.shape]
            Uniformly sampled path.
        expected : array-like, shape=[..., n_times - 1]
            Expected energy at each time step.
        atol : float
            Absolute tolerance.
        """
        res = self.path_energy.energy_per_time(path)
        self.assertAllClose(res, expected, atol=atol)

    def test_dist_from_path_energy_per_time(
        self, point, base_point, n_times, atol=gs.atol
    ):
        """Check the length of a geodesic recovers the distance.

        The length is computed from the energy per time step of the geodesic
        between `base_point` and `point`, uniformly sampled at `n_times`.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
        base_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        n_times : int
            Number of sampled times.
        atol : float
            Absolute tolerance.
        """
        times = gs.linspace(0.0, 1.0, num=n_times)
        geod_points = self.space.metric.geodesic(base_point, end_point=point)(times)

        energy_per_time = self.path_energy.energy_per_time(geod_points)

        delta = 1 / (n_times - 1)
        dist_from_energy_per_time = gs.sum(
            gs.sqrt(energy_per_time * 2 / delta) * delta,
            axis=-1,
        )

        dist = self.space.metric.dist(point, base_point)
        self.assertAllClose(dist_from_energy_per_time, dist, atol=atol)
