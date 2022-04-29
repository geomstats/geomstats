import abc

import matplotlib.pyplot as plt

import geomstats.backend as gs


class Plotter(metaclass=abc.ABCMeta):
    def __init__(self):
        self._belongs = lambda x: True
        self._project = lambda x: x
        self._convert_points = lambda x: x
        self._ax_scale = 1.0

        self._graph_defaults = {
            "scatter": {},
            "plot": {},
        }

    def _create_ax(self):
        if self._dim == 2:
            return plt.subplot()

        return plt.subplot(111, projection="3d")

    def config_ax(self, ax):
        ax_s = self._ax_scale

        if self._dim == 2:
            plt.setp(
                ax,
                xlim=(-ax_s, ax_s),
                ylim=(-ax_s, ax_s),
                xlabel="X",
                ylabel="Y",
            )
        else:
            plt.setp(
                ax,
                xlim=(-ax_s, ax_s),
                ylim=(-ax_s, ax_s),
                zlim=(-ax_s, ax_s),
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
            )

        return ax

    def set_ax(self, ax=None):
        """Set axis."""
        if ax is not None:
            return ax

        ax = self._create_ax()
        return self.config_ax(ax)

    def _check_points(self, points):
        # TODO: points may need to be converted to numpy
        # TODO: check shape to see if only one point?
        if not gs.all(self._belongs(points)):
            raise ValueError("Points do not belong to the space.")

        points = self._project(points)

        return self._convert_points(points)

    def _prepare_vis(
        self,
        ax,
        points,
        space_on,
        grid_on,
        ax_kwargs=None,
        space_kwargs=None,
        grid_kwargs=None,
    ):

        ax_kwargs = ax_kwargs or {}
        ax = self.set_ax(ax=ax, **ax_kwargs)

        if space_on and not grid_on:
            space_kwargs = space_kwargs or {}
            ax = self.plot_space(ax=ax, **space_kwargs)

        if grid_on:
            grid_kwargs = grid_kwargs or {}
            ax = self.plot_grid(ax=ax, **grid_kwargs)

        if points is not None:
            points = self._check_points(points)

        return ax, points

    def _graph(
        self,
        graph_fnc_name,
        points,
        ax=None,
        grid_on=False,
        space_on=False,
        ax_kwargs=None,
        space_kwargs=None,
        grid_kwargs=None,
        after_graph=None,
        **graph_kwargs
    ):
        ax, transformed_points = self._prepare_vis(
            ax,
            points,
            space_on=space_on,
            grid_on=grid_on,
            ax_kwargs=ax_kwargs,
            space_kwargs=space_kwargs,
            grid_kwargs=grid_kwargs,
        )

        graph_fnc = getattr(ax, graph_fnc_name)
        graph_kwargs = _update_dict_with_defaults(
            graph_kwargs, self._graph_defaults.get(graph_fnc_name, {})
        )

        graph_fnc(
            *[transformed_points[..., i] for i in range(self._dim)], **graph_kwargs
        )

        if callable(after_graph):
            after_graph(ax, transformed_points, graph_kwargs)
        elif after_graph is not False:
            self._after_graph(ax, transformed_points, graph_kwargs)

        return ax, transformed_points

    def _after_graph(self, ax, transformed_points, graph_kwargs):
        pass

    def scatter(
        self,
        points,
        ax=None,
        grid_on=False,
        space_on=False,
        ax_kwargs=None,
        space_kwargs=None,
        grid_kwargs=None,
        **scatter_kwargs
    ):
        ax, _ = self._graph(
            "scatter",
            points,
            ax=ax,
            grid_on=grid_on,
            space_on=space_on,
            ax_kwargs=ax_kwargs,
            space_kwargs=space_kwargs,
            grid_kwargs=grid_kwargs,
            **scatter_kwargs
        )
        return ax

    def plot(
        self,
        points,
        ax=None,
        grid_on=False,
        space_on=False,
        ax_kwargs=None,
        space_kwargs=None,
        grid_kwargs=None,
        **plot_kwargs
    ):
        ax, _ = self._graph(
            "plot",
            points,
            ax=ax,
            grid_on=grid_on,
            space_on=space_on,
            ax_kwargs=ax_kwargs,
            space_kwargs=space_kwargs,
            grid_kwargs=grid_kwargs,
            **plot_kwargs
        )
        return ax

    def _get_geodesic_points(
        self,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
        n_points=1000,
    ):

        # TODO: should metric be passed to the space?
        geodesic = self.metric.geodesic(
            initial_point, end_point=end_point, initial_tangent_vec=initial_tangent_vec
        )

        # TODO: check if makes sense for combination initial_point,
        # initial_point_tangent_vec
        t = gs.linspace(0.0, 1.0, n_points)

        return geodesic(t)

    def plot_geodesic(
        self,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
        n_points=1000,
        ax=None,
        space_on=False,
        grid_on=False,
        ax_kwargs=None,
        space_kwargs=None,
        grid_kwargs=None,
        **plot_kwargs
    ):
        """Plot geodesic.

        Follows metric.geodesic signature.
        """
        curve_points = self._get_geodesic_points(
            initial_point, end_point, initial_tangent_vec, n_points
        )

        ax, _ = self._graph(
            "plot",
            curve_points,
            ax=ax,
            grid_on=grid_on,
            space_on=space_on,
            ax_kwargs=ax_kwargs,
            space_kwargs=space_kwargs,
            grid_kwargs=grid_kwargs,
            **plot_kwargs
        )

        return ax

    def plot_inhabitants(self, ax=None):
        return ax

    def plot_vector_field(
        self, tangent_vec, base_point, ax=None, space_on=False, **quiver_kwargs
    ):
        """Draw vectors in the tangent space to sphere at specific base points."""
        return ax

    def plot_tangent_space(self, ax=None):
        return ax

    def plot_space(self, ax=None):
        return ax

    def plot_grid(self, ax=None):
        return ax


def _update_dict_with_defaults(kwargs, kwargs_default):
    for key, value in kwargs_default.items():
        kwargs.setdefault(key, value)

    return kwargs
