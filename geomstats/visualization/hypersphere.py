"""Visualization for Geometric Statistics."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.visualization._plotting import Plotter


class Circle(Plotter):
    """Class used to draw a circle."""

    def __init__(self, n_angles=100):
        super().__init__()

        angles = gs.linspace(0, 2 * gs.pi, n_angles)
        self._circle_x = gs.cos(angles)
        self._circle_y = gs.sin(angles)

        self._space = Hypersphere(dim=1)
        self._metric = self._space.metric
        self._belongs = self._space.belongs

        self._ax_scale = 1.2
        self._dim = 2

        _defaults = {"marker": "o", "linestyle": "None"}
        self._graph_defaults["scatter"] = _defaults
        self._graph_defaults["plot"] = _defaults

    def plot_space(self, ax=None, color="black", **plot_kwargs):
        ax = self.set_ax(ax=ax)
        ax.plot(self._circle_x, self._circle_y, color=color, **plot_kwargs)

        return ax


class Sphere(Plotter):
    """Create the arrays sphere_x, sphere_y, sphere_z to plot a sphere.

    Create the arrays sphere_x, sphere_y, sphere_z of values
    to plot the wireframe of a sphere.
    Their shape is (n_meridians, n_circles_latitude).
    """

    def __init__(self, n_meridians=40, n_circles_latitude=None):
        super().__init__()

        n_circles_latitude = n_circles_latitude or max(n_meridians / 2, 4)

        self._space = Hypersphere(dim=2)
        self._metric = self._space.metric
        self._belongs = self._space.belongs

        u, v = gs.meshgrid(
            gs.arange(0, 2 * gs.pi, 2 * gs.pi / n_meridians),
            gs.arange(0, gs.pi, gs.pi / n_circles_latitude),
        )

        self._center = gs.zeros(3)
        self._radius = 1.0
        self._sphere_x = self._center[0] + self._radius * gs.cos(u) * gs.sin(v)
        self._sphere_y = self._center[1] + self._radius * gs.sin(u) * gs.sin(v)
        self._sphere_z = self._center[2] + self._radius * gs.cos(v)

        self._ax_scale = 1.2
        self._dim = 3

    def config_ax(self, ax):
        ax = super().config_ax(ax)
        ax.set_box_aspect([1.0, 1.0, 1.0])

        return ax

    def _after_graph(self, ax, transformed_points, graph_kwargs):
        return self._add_labels(ax, transformed_points, graph_kwargs)

    def _add_labels(self, ax, points, scatter_kwargs):
        if "label" in scatter_kwargs and len(scatter_kwargs["label"]) == len(points):
            for i_point, point in enumerate(points):
                ax.text(
                    *point,
                    scatter_kwargs["label"][i_point],
                    size=10,
                    zorder=1,
                    color="k",
                )

    def plot_space(self, ax=None, color="grey", alpha=0.2):
        """Plot sphere shape."""
        ax = self.set_ax(ax=ax)

        ax.plot_wireframe(
            self._sphere_x, self._sphere_y, self._sphere_z, color=color, alpha=alpha
        )
        ax.set_box_aspect([1.0, 1.0, 1.0])

        return ax

    def _get_fibonnaci_points(self, n_points=16000):
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

        x_vals = [(self._radius * i) for i in x_vals]
        y_vals = [(self._radius * i) for i in y_vals]
        z_vals = [(self._radius * i) for i in z_vals]

        return gs.array([x_vals, y_vals, z_vals])

    def plot_heatmap(
        self,
        scalar_function,
        ax=None,
        n_points=16000,
        alpha=0.2,
        cmap="jet",
        **scatter_kwargs
    ):
        """Plot a heatmap defined by a loss on the sphere."""
        ax = self.set_ax(ax=ax)

        points = self._get_fibonnaci_points(n_points)
        intensity = gs.array([scalar_function(x) for x in points.T])
        ax.scatter(
            *[points[i, :] for i in range(self._dim)],
            c=intensity,
            alpha=alpha,
            marker=".",
            cmap=plt.get_cmap(cmap),
            **scatter_kwargs,
        )

        return ax
