"""Unit tests for visualization of SPD matrices."""

import matplotlib.pyplot as plt
import tests.conftest
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.visualization import SPDMatricesViz

class TestVisualizationSPDMatrices(tests.conftest.TestCase):
    """Test cases for symmetric_positive_definite.py."""

    def setup_method(self):
        """Set up parameters for the following tests."""
        self.n_samples = 10
        self.spd = SPDMatrices(n=2)
        self.max_z = 1
        self.curr_z = 0.3
        self.hsv = False
        self.spd_viz = SPDMatricesViz(self.max_z)
        plt.figure()

    def test_plot(self):
        """Test case for function plot."""
        self.spd_viz.plot(currZ=self.curr_z)

    def test_plot_grid(self):
        """Test case for function plot_grid."""
        self.spd_viz.plot(currZ=self.curr_z)
        self.spd_viz.plot_grid()

    def test_plot_rendering(self):
        """Test case for function plot_rendering."""
        self.spd_viz.plot(currZ=self.curr_z, hsv=self.hsv)
        self.spd_viz.plot_rendering()

    def test_plot_tangent_space(self):
        """Test case for function plot_tangent_space."""
        self.spd_viz.plot(currZ=self.curr_z)
        self.spd_viz.plot_tangent_space(point=(0, 0, 1))

    def test_scatter(self):
        """Test case for function plot_scatter."""
        self.spd_viz.plot(currZ=self.curr_z)
        self.spd_viz.scatter()

    def test_plot_exp(self):
        """Test case for function plot_exp."""
        self.spd_viz.plot(currZ=self.curr_z)
        self.spd_viz.plot_exp()

    def test_plot_log(self):
        """Test case for function plot_log."""
        self.spd_viz.plot(currZ=self.curr_z)
        self.spd_viz.plot_log()

    def test_plot_geodesic(self):
        """Test case for function plot_geodesic."""
        self.spd_viz.plot(currZ=self.curr_z)
        self.spd_viz.plot_geodesic()
