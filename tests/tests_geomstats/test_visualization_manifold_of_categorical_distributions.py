"""Unit tests for visualization."""

import geomstats.visualization as visualization
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tests.conftest

from CategoricalDistributionsManifold import CategoricalDistributionsManifold

matplotlib.use("Agg")  # NOQA


class TestVisualizationManifoldOfCategoricalDistributions(
        tests.conftest.TestCase):
    def setup_method(self):
        self.n_samples = 10
        self.CD2 = CategoricalDistributionsManifold(dim=2)
        self.CD3 = CategoricalDistributionsManifold(dim=3)
        plt.figure()

    @staticmethod
    def test_tutorial_matplotlib():
        visualization.tutorial_matplotlib()

    def test_plot(self):
        self.CD2.plot()
        self.CD3.plot()

    def test_scatter(self):
        self.CD2.scatter(self.n_samples)
        self.CD3.scatter(self.n_samples)

    def test_plot_geodesic(self):
        b2 = np.array([0.2, 0.3, 0.5])
        v2 = np.array([1, 0, 0])
        self.CD2.plot_geodesic(initial_point=b2, tangent_vector=v2)

        b3 = np.array([0.3, 0.3, 0.2, 0.2])
        v3 = np.array([1, 0, 0, 0])
        self.CD3.plot_geodesic(initial_point=b3, tangent_vector=v3)

    def test_plot_grid(self):
        self.CD2.plot_grid()

    @staticmethod
    def teardown_method():
        plt.close()
