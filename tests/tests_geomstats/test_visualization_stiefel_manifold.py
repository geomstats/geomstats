"""Stiefel manifold visualization test module."""
# import geomstats.backend as gs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geomstats.geometry.stiefel import Stiefel

import tests.conftest
from geomstats.visualization.stiefel_manifold import Arrow2D, StiefelCircle, StiefelSphere

matplotlib.use("Agg")  # NOQA


class TestStiefelManifold(tests.conftest.TestCase):
    """Class to test all the functions for S(2,2)."""

    def setup_method(self):
        """Initialize variables."""
        self.n_samples = 10
        self.v21 = Stiefel(2, 1)
        self.v22 = Stiefel(2, 2)
        self.v31 = Stiefel(3, 1)
        self.St_sph = StiefelSphere()
        self.St_cir = StiefelCircle()

        self.sph_ax = self.St_sph.set_ax(ax=None)
        self.cir_ax = self.St_cir.set_ax(ax=None)

        plt.figure()

    def test_stiefel_sphere_set_ax(self):
        """Set ax for sphere."""
        self.St_sph.set_ax(ax=None)

    def test_stiefel_sphere_set_view(self):
        """Set view for sphere."""
        self.St_sph.set_view()

    def test_stiefel_sphere_draw(self):
        """Draw a stiefel manifold."""
        self.St_sph.draw()

    def test_stiefel_sphere_coordinates_transformation(self):
        """Makes random points from distribution."""
        points = self.v31.random_uniform(10)[2, :, 0]
        self.St_sph.coordinates_transformation(points)

    def test_stiefel_sphere_add_points(self):
        """Take, add points from random distribution on manifold, visualize."""
        points = self.v31.random_uniform(10)
        self.St_sph.add_points(points)
        self.St_sph.draw_points(self.sph_ax)

    def test_stiefel_sphere_draw_points(self):
        """Take points from random distribution on manifold and visualize."""
        points = self.v31.random_uniform(10)
        self.St_sph.draw_points(points)

    def test_stiefel_sphere_clear_points(self):
        """Clear points, draw points to verify points were cleared."""
        self.St_sph.clear_points()
        self.St_sph.draw_points(self.sph_ax)

    def test_stiefel_sphere_draw_mesh(self):
        """Take point from random distribution and draw mesh."""
        point = self.v31.random_uniform(10)[2, :, 0]
        self.St_sph.draw_mesh(point)

    def test_stiefel_circle_add_points(self):
        """Add a point to stiefel circle."""
        points = self.v21.random_uniform(100)[:, :, 0]
        self.St_cir.add_points(points)
        self.St_cir.draw(self.cir_ax, markersize=5, alpha=0.5, label="points")
        self.St_cir.clear_points()

    def test_stiefel_circle_draw_points(self):
        """Draw a point on stiefel circle."""
        points = self.v21.random_uniform(100)[:, :, 0]
        self.St_cir.draw_points(self.cir_ax, points)

    def test_stiefel_circle_draw_line_to_point(self):
        """Generate a vector on a random point."""
        point = self.v22.random_uniform(1)
        if np.linalg.det(point) > 0:
            v_1 = point[:, 1]
            p_1 = point[:, 0]
        else:
            v_1 = point[:, 0]
            p_1 = point[:, 1]

        self.St_cir.draw_line_to_point(ax=self.cir_ax, point=p_1, line=v_1)

    def test_stiefel_circle_draw_curve(self):
        """Draw a curve."""
        group1_points = self.v22.random_uniform(2)
        det = np.linalg.det(group1_points)
        group1_points_to_circle = np.zeros((2, 2, 2))
        for i in range(2):
            if det[i] > 0:
                group1_points_to_circle[i, :, 0] = group1_points[i][:, 0]
                group1_points_to_circle[i, :, 1] = group1_points[i][:, 1]
            elif det[i] < 0:
                group1_points_to_circle[i, :, 0] = group1_points[i][:, 1]
                group1_points_to_circle[i, :, 1] = group1_points[i][:, 0]

        self.St_cir.add_points(group1_points_to_circle[:, :, 0])
        self.St_cir.draw_curve(color="b", lw=3, label="geodesic line")
        self.St_cir.draw_curve()

    def test_stifel_circle_draw_tangent_space(self):
        """Drawing tangent space of a random point."""
        p_1 = self.v22.random_uniform(1)
        if np.linalg.det(p_1) > 0:
            p_1 = p_1[:, 0]
        else:
            p_1 = p_1[:, 1]
        self.St_cir.draw_tangent_space(ax=self.cir_ax, base_point=p_1)

    def test_stiefel_circle_plot_rendering(self):
        """Test drawing the manifold with regularly sampled data."""
        self.St_cir.plot_rendering(self.v21, 100)
        self.St_cir.plot_rendering(self.v22, 100)

    def test_stiefel_circle_plot_geodesic(self):
        """Test visualizing a (discretised) geodesic."""
        ini_point = self.v22.random_uniform(1)
        end_point = self.v22.random_uniform(1)
        self.St_cir.plot_geodesic(ini_point, end_point)

    def test_stifel_arrow2d_draw(self):
        """Draw 2d arrow."""
        p_1 = self.v22.random_uniform(1)
        if np.linalg.det(p_1) > 0:
            v_1 = p_1[:, 1]
            p_1 = p_1[:, 0]
        else:
            v_1 = p_1[:, 0]
            p_1 = p_1[:, 1]
        arrow = Arrow2D(point=p_1, vector=v_1)
        ax = self.St_cir.set_ax(ax=None)
        arrow.draw(ax=ax)

    @staticmethod
    def teardown_method():
        """Close the plot."""
        plt.close()
