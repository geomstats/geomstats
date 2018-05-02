"""Unit tests for visualization module."""

import matplotlib
matplotlib.use('Agg')  # NOQA
import unittest

import geomstats.visualization as visualization
from geomstats.hypersphere import Hypersphere
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


SO3_GROUP = SpecialOrthogonalGroup(n=3)
SE3_GROUP = SpecialEuclideanGroup(n=3)
S2 = Hypersphere(dimension=2)


# TODO(nina): add tests for examples

class TestVisualizationMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.n_samples = 10

    def test_plot_points_so3(self):
        points = SO3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SO3_GROUP')

    def test_plot_points_se3(self):
        points = SE3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SE3_GROUP')

    def test_plot_points_s2(self):
        points = S2.random_uniform(self.n_samples)
        visualization.plot(points, space='S2')


if __name__ == '__main__':
        unittest.main()
