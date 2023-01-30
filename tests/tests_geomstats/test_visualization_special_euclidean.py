"""Unit tests for visualization."""

import geomstats.backend as gs
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

        plt.figure()

    def test_plot_points_se2(self):
        points = self.SE2_VEC.random_point(self.n_samples)
        points_mat = self.SE2_VEC.matrix_from_vector(points)
        visualization.plot(points_mat, space="SE2_GROUP")

    def test_plot_geodesic_se2(self):
        METRIC = self.SE2_VEC.left_canonical_metric
        initial_point = self.SE2_VEC.identity
        initial_tangent_vec = gs.array([1.8, 0.2, 0.3])
        N_STEPS = 40
        visualization.SpecialEuclidean2.plot_geodesic(
            initial_point, initial_tangent_vec, METRIC, N_STEPS
        )

    def test_plot_points_se3(self):
        points = self.SE3_VEC.random_point(self.n_samples)
        print(points.shape)
        visualization.plot(points, space="SE3_GROUP")

    def test_plot_geodesic_se3(self):
        METRIC = self.SE3_VEC.left_canonical_metric
        initial_point = self.SE3_VEC.identity
        initial_tangent_vec = gs.array([1.8, 0.2, 0.3, 3.0, 3.0, 1.0])
        N_STEPS = 40
        visualization.SpecialEuclidean3.plot_geodesic(
            initial_point, initial_tangent_vec, METRIC, N_STEPS
        )
        
