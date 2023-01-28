"""Unit tests for visualization."""
import matplotlib
import matplotlib.pyplot as plt
import tests.conftest
from grassmannian_manifold.grassmannian import Grassmannian

matplotlib.use("Agg")  # NOQA


class TestVisualizationGrassmannian(tests.conftest.TestCase):
    """
    Test class specifically to test Grassmannian
    manifold by plotting 2D and 3D figure
    """
    def setup_method(self):
        """
        Prepares a Grassmannian object for 2D and 3D plotting
         that uses 10 samples in the figure.
        """
        self.n_samples = 10
        self.grassmannian21 = Grassmannian(2, 1)
        self.grassmannian31 = Grassmannian(3, 1)
        plt.figure()

    def test_plot_3d(self):
        """Test to see if the plot function return a 3d plot
        with grid existing or not based on the boolean value
        passed onto the function.
        """
        # points = self.grassmannian31.random_uniform(self.n_samples)
        self.grassmannian31.plot(True)

    # def test_plot_3d(self):
    #     self.tick= False
    #     points = self.grassmannian31.random_uniform(self.n_samples)
    #     self.grassmannian31.plot(False)
    def test_plot_rendering_3d(self):
        """ "Test to see if the plot function returns a 3d plot
        with a manifold that has a regularly sampled data."""
        # self.tick = True
        # points = self.grassmannian31.random_uniform(self.n_samples)
        self.grassmannian31.plot(True)

    # def test_plot_rendering_3d(self):
    #     self.tick= False
    #     points = self.grassmannian31.random_uniform(self.n_samples)
    #     self.grassmannian31.plot(False)

    @staticmethod
    def teardown_method():
        plt.close()
