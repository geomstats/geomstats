"""
Unit tests for visualization.
"""

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt

import geomstats.tests
import geomstats.visualization as visualization

from geomstats.hyperbolic_space import HyperbolicSpace
from geomstats.hypersphere import Hypersphere
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


class TestVisualizationMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.n_samples = 10
        self.SO3_GROUP = SpecialOrthogonalGroup(n=3)
        self.SE3_GROUP = SpecialEuclideanGroup(n=3)
        self.S1 = Hypersphere(dimension=1)
        self.S2 = Hypersphere(dimension=2)
        self.H2 = HyperbolicSpace(dimension=2)

        plt.figure()

    @geomstats.tests.np_only
    def test_plot_points_so3(self):
        points = self.SO3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SO3_GROUP')

    @geomstats.tests.np_only
    def test_plot_points_se3(self):
        points = self.SE3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SE3_GROUP')

    @geomstats.tests.np_only
    def test_plot_points_s1(self):
        points = self.S1.random_uniform(self.n_samples)
        visualization.plot(points, space='S1')

    @geomstats.tests.np_only
    def test_plot_points_s2(self):
        points = self.S2.random_uniform(self.n_samples)
        visualization.plot(points, space='S2')

    @geomstats.tests.np_only
    def test_plot_points_h2_poincare_disk(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_disk')

    @geomstats.tests.np_only
    def test_plot_points_h2_poincare_half_plane(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_half_plane')

    @geomstats.tests.np_only
    def test_plot_points_h2_klein_disk(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_klein_disk')


if __name__ == '__main__':
        geomstats.tests.main()
