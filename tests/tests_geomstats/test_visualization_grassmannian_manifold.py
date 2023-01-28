"""Unit tests for visualization."""
import os
import sys
import warnings

sys.path.append(os.path.dirname(os.getcwd()))
# TODO: Change these paths with the corresponding paths on your computer


print(os.getcwd())
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from grassmannian_manifold.grassmannian import Grassmannian
import tests.conftest
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonalMatrices,
)

matplotlib.use("Agg")  # NOQA


class TestVisualizationGrassmanian(tests.conftest.TestCase):
    def setup_method(self):
        
        self.n_samples = 10
        self.grassmannian21 = Grassmannian(2, 1)
        self.grassmannian31 = Grassmannian(3, 1)
        plt.figure()

    def test_plot_2d(self): 
        """
        Test the plotting function of Grassmannian for 2 dimensions 
        """
        points = self.grassmannian21.random_uniform(self.n_samples)
        self.grassmannian21.plot(True)

    def test_plot_2d_render(self): 
        """
        Test the plotting function of Grassmannian for 2 dimensions 
        when it's regularly sampled
        """
        points = self.grassmannian21.random_uniform(self.n_samples)
        self.grassmannian21.plot(True)


 

    @staticmethod
    def teardown_method():
        plt.close()