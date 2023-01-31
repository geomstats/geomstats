"""Unit tests for visualization."""

import geomstats.visualization as visualization
import matplotlib
import matplotlib.pyplot as plt
import tests.conftest
from geomstats.geometry.special_euclidean import SpecialEuclidean

matplotlib.use("Agg")  # NOQA


class TestVisualization(tests.conftest.TestCase):
    """Test of visualization for Special Euclidean groups."""

    def setup_method(self):
        self.n_samples = 10
        self.SE2_VEC = SpecialEuclidean(n=2, point_type="vector")
        self.SE3_VEC = SpecialEuclidean(n=3, point_type="vector")
        self.viz_SE2 = visualization.SpecialEuclidean2()
        self.viz_SE3 = visualization.SpecialEuclidean3()

        plt.figure()

    def test_plot_points_se2(self):
        points = self.SE2_VEC.random_point(n_samples=self.n_samples)
        visualization.plot(points, space="SE2_GROUP")

    def test_plot_geodesic_se2(self):
        initial_point = self.SE2_VEC.random_point(n_samples=1)
        initial_tangent_vec = self.SE2_VEC.random_tangent_vec(base_point=initial_point)
        N_STEPS = 10
        self.viz_SE2.plot_geodesic(
            point=initial_point, vector=initial_tangent_vec, n_steps=N_STEPS
        )

    def test_plot_points_se3(self):
        points = self.SE3_VEC.random_point(n_samples=self.n_samples)
        print(points.shape)
        visualization.plot(points, space="SE3_GROUP")

    def test_plot_geodesic_se3(self):
        initial_point = self.SE3_VEC.random_point(n_samples=1)
        initial_tangent_vec = self.SE3_VEC.random_tangent_vec(base_point=initial_point)
        N_STEPS = 10
        self.viz_SE3.plot_geodesic(
            point=initial_point, vector=initial_tangent_vec, n_steps=N_STEPS
        )
