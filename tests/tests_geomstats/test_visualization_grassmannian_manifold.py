"""Unit tests for visualization."""
import os
import sys
import warnings
import matplotlib
import matplotlib.pyplot as plt
from geomstats.visualization.grassmannian import Grassmannian
import tests.conftest

sys.path.append(os.path.dirname(os.getcwd()))
# TODO: Change these paths with the corresponding paths on your computer

print(os.getcwd())
warnings.filterwarnings("ignore")

matplotlib.use("Agg")  # NOQA


class TestVisualizationGrassmanian(tests.conftest.TestCase):
    def setup_method(self):
        """Set up for testing Grassmannian manifold.
        """
        self.n_samples = 10
        self.grassmannian21 = Grassmannian(2, 1)
        self.grassmannian31 = Grassmannian(3, 1)
        plt.figure()

    def test_plot_2d(self):
        """Test the plotting function of Grassmannian for 2 dimensions.
        """
        self.grassmannian21.plot(True)

    def test_plot_2d_render(self):
        """Test plot rendering function for 2D.

        Test the plotting function of Grassmannian for 2 dimensions
        when it's regularly sampled
        """
        self.grassmannian21.plot_rendering(True)

    @staticmethod
    def teardown_method():

        plt.close()
