"""Unit tests for visualization."""

import matplotlib.pyplot as plt
import tests.conftest
from geomstats.geometry.spd_matrices import SPDMatrices
import symmetric_positive_definite


class TestVisualizationSPD(tests.conftest.TestCase):
    """Test cases for symmetric_positive_definite.py."""

    def setup_method(self):
        """Set up parameters for the following tests."""
        self.n_samples = 10
        self.spd = SPDMatrices(n=2)
        self.max_z = 1
        self.curr_z = 0.3
        self.hsv = False
        plt.figure()

    def test_plot(self):
        """Test case for function plot."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z)

    def test_plot_grid(self):
        """Test case for function plot_grid."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z)
        viz.plot_grid()

    def test_plot_rendering(self):
        """Test case for function plot_rendering."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z, hsv=self.hsv)
        viz.plot_rendering()

    def test_plot_tangent_space(self):
        """Test case for function plot_tangent_space."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z)
        viz.plot_tangent_space(point=(0, 0, 1))

    def test_scatter(self):
        """Test case for function plot_scatter."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z)
        viz.scatter()

    def test_plot_exp(self):
        """Test case for function plot_exp."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z)
        viz.plot_exp()

    def test_plot_log(self):
        """Test case for function plot_log."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z)
        viz.plot_log()

    def test_plot_geodesic(self):
        """Test case for function plot_geodesic."""
        viz = symmetric_positive_definite.SymPosDefVisualization(self.max_z)
        viz.plot(currZ=self.curr_z)
        viz.plot_geodesic()
