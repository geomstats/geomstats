"""Unit tests for visualization."""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

import geomstats.backend as gs
import tests.conftest
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.visualization.beta_distributions import Beta

sys.path.append(os.path.dirname(os.getcwd()))

matplotlib.use("Agg")  # NOQA


class TestVisualizationBeta(tests.conftest.TestCase):
    def setup_method(self):
        self.n_samples = 10
        self.Beta = BetaDistributions()
        self.beta_viz = Beta()

    def test_plot_beta(self):
        points = gs.random.rand(2, 2)
        self.beta_viz.plot(points)

    def test_scatter_beta(self):
        num_points = gs.random.randint(2, 50)
        points = gs.random.rand(num_points, 2)
        self.beta_viz.plot(points)

    def test_plot_geodesic_ball(self):
        center = gs.random.rand(1, 2)
        n_rays = gs.random.randint(2, 100)
        ray_length = 1 - gs.random.uniform(0.1, 1)
        self.beta_viz.plot_geodestic_ball(center, n_rays, ray_length)

    def test_plot_vector_field(self):
        center = gs.random.rand(1, 2)
        num_vec = gs.random.randint(1, 20)
        tan_vec = gs.array(
            [[gs.random.uniform(-1, 1) for i in range(2)] for j in range(num_vec)]
        )
        ray_length = 1 - gs.random.uniform(0.1, 1)
        self.beta_viz.plot_vector_field(center, tan_vec, ray_length)

    def test_plot_grid(self):
        size = gs.array([gs.random.randint(1, 6) for i in range(2)])
        initial_point = gs.array([gs.random.uniform(0, 1) for i in range(2)])
        n_steps = 100
        n_points = gs.random.randint(1, 15)
        step = gs.random.uniform(0, 2)
        self.beta_viz.plot_grid(size, initial_point, n_steps, n_points, step)

    def test_plot_rendering(self):
        initial_point = gs.array([gs.random.uniform(0, 1) for i in range(2)])
        size = gs.array([gs.random.randint(1, 8) for i in range(2)])
        sampling_period = gs.random.uniform(0.1, 15)

        self.beta_viz.plot_rendering(initial_point, size, sampling_period)

    def test_plot_geodesic(self):
        n_steps = 100
        n_points = gs.random.randint(20, 50)
        cc = gs.zeros((n_points, 3))
        cc[:, 2] = gs.linspace(0, 1, n_points)
        point_a = gs.array([gs.random.uniform(0, 10) for i in range(2)])
        point_b = gs.array([gs.random.uniform(0, 10) for i in range(2)])

        self.beta_viz.plot_geodesic(
            initial_point=point_a,
            end_point=point_b,
            n_points=n_points,
            color=cc,
            n_steps=n_steps,
        )

        tangent_vector = gs.array([gs.random.uniform(-1, 1), gs.random.uniform(-1, 1)])
        self.beta_viz.plot_geodesic(
            initial_point=point_a,
            initial_tangent_vec=tangent_vector,
            n_points=n_points,
            color=cc,
            n_steps=n_steps,
        )

    @staticmethod
    def teardown_method():
        plt.close()
