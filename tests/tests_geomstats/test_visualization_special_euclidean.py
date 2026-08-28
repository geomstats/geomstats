"""Unit tests for visualization of SE(2) and SE(3)."""

import matplotlib
import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
import tests.conftest
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.visualization.special_euclidean import (
    SpecialEuclidean2,
    SpecialEuclidean3,
)

matplotlib.use("Agg")  # NOQA


class TestSpecialEuclidean2(tests.conftest.TestCase):
    """Test of visualization for SE(2) group."""

    def setup_method(self):
        self.n_samples = 10
        self.SE2_VEC = SpecialEuclidean(n=2, point_type="vector")
        plt.figure()

    @staticmethod
    def test_set_ax():
        SpecialEuclidean2.set_ax()

    def test_add_points(self):
        test_points = self.SE2_VEC.random_point(self.n_samples)
        viz = SpecialEuclidean2()
        viz.add_points(test_points)

    def test_draw_points(self):
        points = self.SE2_VEC.random_point(self.n_samples)
        points_mat = self.SE2_VEC.matrix_from_vector(points)
        visualization.plot(points_mat, space="SE2_GROUP")

    def test_plot_geodesic(self):
        metric = self.SE2_VEC.left_canonical_metric
        initial_point = self.SE2_VEC.identity
        initial_tangent_vec = gs.array([1.8, 0.2, 0.3])
        n_steps = 40
        SpecialEuclidean2.plot_geodesic(
            initial_point, initial_tangent_vec, metric, n_steps
        )


class TestSpecialEuclidean3(tests.conftest.TestCase):
    """Test of visualization for SE(3) group."""

    def setup_method(self):
        self.n_samples = 10
        self.SE3_VEC = SpecialEuclidean(n=3, point_type="vector")
        plt.figure()

    @staticmethod
    def test_set_ax():
        SpecialEuclidean3.set_ax()

    def test_add_points(self):
        test_points = self.SE3_VEC.random_point(self.n_samples)
        viz = SpecialEuclidean3()
        viz.add_points(test_points)

    def test_draw_points(self):
        points = self.SE3_VEC.random_point(self.n_samples)
        visualization.plot(points, space="SE3_GROUP")

    def test_plot_geodesic(self):
        metric = self.SE3_VEC.left_canonical_metric
        initial_point = self.SE3_VEC.identity
        initial_tangent_vec = gs.array([1.8, 0.2, 0.3, 3.0, 3.0, 1.0])
        n_steps = 40
        SpecialEuclidean3.plot_geodesic(
            initial_point, initial_tangent_vec, metric, n_steps
        )
