"""
Unit tests for visualization.
"""

import matplotlib
matplotlib.use('Agg')  # NOQA
import unittest

import geomstats.visualization as visualization

from geomstats.hyperbolic_space import HyperbolicSpace
from geomstats.hypersphere import Hypersphere
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


# TODO(nina): add tests for examples

class TestVisualizationMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.n_samples = 10
        self.SO3_GROUP = SpecialOrthogonalGroup(n=3)
        self.SE3_GROUP = SpecialEuclideanGroup(n=3)
        self.S2 = Hypersphere(dimension=2)
        self.H2 = HyperbolicSpace(dimension=2)

    def test_plot_points_so3(self):
        points = self.SO3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SO3_GROUP')

    def test_plot_points_se3(self):
        points = self.SE3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SE3_GROUP')

    def test_plot_points_s2(self):
        points = self.S2.random_uniform(self.n_samples)
        visualization.plot(points, space='S2')

    def test_plot_points_h2_poincare_disk(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_disk')

    def test_plot_points_h2_poincare_half_plane(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_half_plane')

    def test_plot_points_h2_klein_disk(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_klein_disk')


if __name__ == '__main__':
        unittest.main()
