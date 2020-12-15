"""Unit tests for visualization."""

import matplotlib
import matplotlib.pyplot as plt

import geomstats.tests
import geomstats.visualization as visualization
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

matplotlib.use('Agg')  # NOQA


class TestVisualization(geomstats.tests.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.SO3_GROUP = SpecialOrthogonal(n=3, point_type='vector')
        self.SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
        self.S1 = Hypersphere(dim=1)
        self.S2 = Hypersphere(dim=2)
        self.H2 = Hyperbolic(dim=2)
        self.H2_half_plane = PoincareHalfSpace(dim=2)

        plt.figure()

    @staticmethod
    def test_tutorial_matplotlib():
        visualization.tutorial_matplotlib()

    def test_plot_points_so3(self):
        points = self.SO3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SO3_GROUP')

    def test_plot_points_se3(self):
        points = self.SE3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space='SE3_GROUP')

    @geomstats.tests.np_and_pytorch_only
    def test_plot_points_s1(self):
        points = self.S1.random_uniform(self.n_samples)
        visualization.plot(points, space='S1')

    def test_plot_points_s2(self):
        points = self.S2.random_uniform(self.n_samples)
        visualization.plot(points, space='S2')

    def test_plot_points_h2_poincare_disk(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_disk')

    def test_plot_points_h2_poincare_half_plane_ext(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_half_plane',
                           point_type='extrinsic')

    def test_plot_points_h2_poincare_half_plane_none(self):
        points = self.H2_half_plane.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_half_plane')

    def test_plot_points_h2_poincare_half_plane_hs(self):
        points = self.H2_half_plane.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_poincare_half_plane',
                           point_type='half_space')

    def test_plot_points_h2_klein_disk(self):
        points = self.H2.random_uniform(self.n_samples)
        visualization.plot(points, space='H2_klein_disk')

    @staticmethod
    def test_plot_points_se2():
        points = SpecialEuclidean(n=2, point_type='vector').random_uniform(4)
        visu = visualization.SpecialEuclidean2(points, point_type='vector')
        ax = visu.set_ax()
        visu.draw(ax)
