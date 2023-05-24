"""Unit tests for visualization of Klein bottle manifold."""

import geomstats.visualization as visualization
import matplotlib
import matplotlib.pyplot as plt
import tests.conftest
from geomstats.geometry.klein_bottle import KleinBottle

matplotlib.use("Agg")  # NOQA


class TestVisualizationKleinBottle(tests.conftest.TestCase):
    """Class used to test Klein Bottle visualization."""

    def setup_method(self):
        """Set up figure for Klein Bottle visualization."""
        self.n_samples = 10
        self.klein_bottle = KleinBottle(equip=True)
        self.viz_klein_bottle = visualization.klein_bottle()

        plt.figure()

    def test_draw_points_kb(self):
        """Test drawing of 2D point cloud data."""
        points = self.klein_bottle.random_point(self.n_samples)
        self.viz_klein_bottle.add_points(points)
        self.viz_klein_bottle.draw_points(space="KB")
        self.viz_klein_bottle.clear_points(self)

    def test_plot_kb(self):
        """Test plotting of Klein Bottle visualization."""
        points = self.klein_bottle.random_point(self.n_samples)
        self.viz_klein_bottle.add_points(points)
        self.viz_klein_bottle.plot(self, coords_type="intrinsic")
        self.viz_klein_bottle.clear_points(self)
