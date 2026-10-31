"""Unit tests for visualization."""

import geomstats.backend as gs
import geomstats.visualization as visualization
import matplotlib.pyplot as plt
import tests.conftest
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_linear import SpecialLinear
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestVisualization(tests.conftest.TestCase):
    """Class for testing special linear visualization."""

    def setup_method(self):
        """Choose group."""
        self.n_samples = 10
        self.SO3_GROUP = SpecialOrthogonal(n=3, point_type="vector")
        self.SE3_GROUP = SpecialEuclidean(n=3, point_type="vector")
        self.S1 = Hypersphere(dim=1)
        self.S2 = Hypersphere(dim=2)
        self.H2 = Hyperbolic(dim=2)
        self.H2_half_plane = PoincareHalfSpace(dim=2)
        self.M32 = Matrices(m=3, n=2)
        self.S32 = PreShapeSpace(k_landmarks=3, m_ambient=2)
        self.KS = visualization.KendallSphere()
        self.M22 = Matrices(m=2, n=2)
        self.M33 = Matrices(m=3, n=3)
        self.S33 = PreShapeSpace(k_landmarks=3, m_ambient=3)
        self.KD = visualization.KendallDisk()
        self.SPD = SPDMatrices(n=2)
        self.SL2 = SpecialLinear(n=2)
        self.SL3 = SpecialLinear(n=3)

        plt.figure()

    def test_plot_points_SL2(self):
        """Animate the SL2(R) group.

        Plot animated transformation of a 2D grid under the action
        of a geodesic between two random 2x2 elements of the
        special linear group
        """
        base_point = self.M22.random_point()
        final_point = self.M22.random_point()
        geodesicspd = self.M22.metric.geodesic(
            initial_point=base_point, end_point=final_point
        )
        points = geodesicspd(gs.linspace(0.0, 1.0, 20))
        projected_points = self.SL2.projection(points)
        visualization.plot(projected_points, ax=None, space="SL2")

    def test_plot_points_SL3(self):
        """Animate the SL3(R) group.

        Plot animated transformation of a cube under the action
        of a geodesic between two random 3x3 elements of the
        special linear group
        """
        base_point = self.M33.random_point()
        final_point = self.M33.random_point()
        geodesicspd = self.M33.metric.geodesic(
            initial_point=base_point, end_point=final_point
        )
        points = geodesicspd(gs.linspace(0.0, 1.0, 20))
        projected_points = self.SL3.projection(points)
        visualization.plot(projected_points, ax=None, space="SL3")

    @staticmethod
    def teardown_method():
        """Close plots."""
        plt.close()
