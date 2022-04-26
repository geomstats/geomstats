

import abc


import geomstats.backend as gs

# TODO: add args to methods


class Plotter(metaclass=abc.ABCMeta):

    def set_ax(self, ax=None):
        """Set axis."""
        if ax is not None:
            return ax

        ax = self._create_ax()
        return self.config_ax(ax)

    def _check_points(self, points):
        # TODO: points may need to be converted to numpy
        if not gs.all(self._belongs(points)):
            raise ValueError("Points do not belong to the space.")

        return points

    def _prepare_vis(self, ax, points, space_on, grid_on):
        ax = self.set_ax(ax=ax)
        points = self._check_points(points)

        if space_on:
            self.plot_space(ax=ax)

        if grid_on:
            self.plot_grid(ax=ax)

        return ax, points

    def scatter(self, grid_on=False, space_on=False, **scatter_kwargs):
        pass

    def plot(self, grid_on=False, space_on=False, **plot_kwargs):
        pass

    def plot_geodesic(self):
        pass

    def plot_inhabitants(self):
        pass

    def plot_vector_field(self):
        pass

    def plot_tangent_space(self):
        pass

    def plot_space(self, ax=None):
        pass

    def plot_grid(self):
        pass
