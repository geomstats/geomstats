"""Visualization for Geometric Statistics."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere

S1 = Hypersphere(dim=1)
S2 = Hypersphere(dim=2)

AX_SCALE = 1.2


class Circle:
    """Class used to draw a circle."""

    def __init__(self, n_angles=100, points=None):
        angles = gs.linspace(0, 2 * gs.pi, n_angles)
        self.circle_x = gs.cos(angles)
        self.circle_y = gs.sin(angles)
        self.points = []
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        """Set axis."""
        if ax is None:
            ax = plt.subplot()
        ax_s = AX_SCALE
        plt.setp(
            ax,
            xlim=(-ax_s, ax_s),
            ylim=(-ax_s, ax_s),
            xlabel="X",
            ylabel="Y",
        )
        return ax

    def add_points(self, points):
        """Add points."""
        if not gs.all(S1.belongs(points)):
            raise ValueError("Points do  not belong to the circle.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw(self, ax, **plot_kwargs):
        """Plot circle shape."""
        ax.plot(self.circle_x, self.circle_y, color="black")
        if self.points:
            self.draw_points(ax, **plot_kwargs)

    def draw_points(self, ax, points=None, **plot_kwargs):
        """Plot points."""
        if points is None:
            points = self.points
        points = gs.array(points)
        ax.plot(points[:, 0], points[:, 1], marker="o", linestyle="None", **plot_kwargs)

    def plot(self, points, ax=None, **point_draw_kwargs):
        """Plot points in the circle."""
        ax = self.set_ax(ax=ax)
        self.add_points(points)
        self.draw(ax, **point_draw_kwargs)


class Sphere:
    """Create the arrays sphere_x, sphere_y, sphere_z to plot a sphere.

    Create the arrays sphere_x, sphere_y, sphere_z of values
    to plot the wireframe of a sphere.
    Their shape is (n_meridians, n_circles_latitude).
    """

    def __init__(self, n_meridians=40, n_circles_latitude=None, points=None):
        if n_circles_latitude is None:
            n_circles_latitude = max(n_meridians / 2, 4)

        u, v = gs.meshgrid(
            gs.arange(0, 2 * gs.pi, 2 * gs.pi / n_meridians),
            gs.arange(0, gs.pi, gs.pi / n_circles_latitude),
        )

        self.center = gs.zeros(3)
        self.radius = 1
        self.sphere_x = self.center[0] + self.radius * gs.cos(u) * gs.sin(v)
        self.sphere_y = self.center[1] + self.radius * gs.sin(u) * gs.sin(v)
        self.sphere_z = self.center[2] + self.radius * gs.cos(v)

        self.points = []
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None):
        """Set axis."""
        if ax is None:
            ax = plt.subplot(111, projection="3d")

        ax_s = AX_SCALE
        plt.setp(
            ax,
            xlim=(-ax_s, ax_s),
            ylim=(-ax_s, ax_s),
            zlim=(-ax_s, ax_s),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        ax.set_box_aspect([1.0, 1.0, 1.0])
        return ax

    def add_points(self, points):
        """Add points."""
        if not gs.all(S2.belongs(points)):
            raise ValueError("Points do not belong to the sphere.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw(self, ax, **scatter_kwargs):
        """Plot sphere shape."""
        ax.plot_wireframe(
            self.sphere_x, self.sphere_y, self.sphere_z, color="grey", alpha=0.2
        )
        ax.set_box_aspect([1.0, 1.0, 1.0])
        if self.points:
            self.draw_points(ax, **scatter_kwargs)

    def draw_points(self, ax, points=None, **scatter_kwargs):
        """Plot points."""
        if points is None:
            points = self.points
        points = [gs.autodiff.detach(point) for point in points]
        points = [gs.to_numpy(point) for point in points]
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        points_z = [point[2] for point in points]
        ax.scatter(points_x, points_y, points_z, **scatter_kwargs)

        for i_point, point in enumerate(points):
            if "label" in scatter_kwargs:
                if len(scatter_kwargs["label"]) == len(points):
                    ax.text(
                        point[0],
                        point[1],
                        point[2],
                        scatter_kwargs["label"][i_point],
                        size=10,
                        zorder=1,
                        color="k",
                    )

    def get_fibonnaci_points(self, n_points=16000):
        """Get spherical Fibonacci point sets.

        Point sets are yield nearly uniform point distributions on the unit
        sphere.
        """
        x_vals = []
        y_vals = []
        z_vals = []

        offset = 2.0 / n_points
        increment = gs.pi * (3.0 - gs.sqrt(5.0))

        for i in range(n_points):
            y = ((i * offset) - 1) + (offset / 2)
            r = gs.sqrt(1 - pow(y, 2))

            phi = ((i + 1) % n_points) * increment

            x = gs.cos(phi) * r
            z = gs.sin(phi) * r

            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

        x_vals = [(self.radius * i) for i in x_vals]
        y_vals = [(self.radius * i) for i in y_vals]
        z_vals = [(self.radius * i) for i in z_vals]

        return gs.array([x_vals, y_vals, z_vals])

    def plot_heatmap(self, ax, scalar_function, n_points=16000, alpha=0.2, cmap="jet"):
        """Plot a heatmap defined by a loss on the sphere."""
        points = self.get_fibonnaci_points(n_points)
        intensity = gs.array([scalar_function(x) for x in points.T])
        ax.scatter(
            points[0, :],
            points[1, :],
            points[2, :],
            c=intensity,
            alpha=alpha,
            marker=".",
            cmap=plt.get_cmap(cmap),
        )

    def plot(self, points, ax=None, **point_draw_kwargs):
        """Plot points in the sphere."""
        ax = self.set_ax(ax=ax)
        self.points = []
        self.add_points(points)
        self.draw(ax, **point_draw_kwargs)
