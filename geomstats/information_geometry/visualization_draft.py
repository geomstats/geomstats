"""Draft of visualizer (plotter) for information geometry."""

import geomstats.backend as gs


class Visualizer2D:
    """Visualizer for 2-D parameter distributions."""

    def __init__(self, space):
        self.space = space

    def scatter(self, ax, point, **kwargs):
        """Plot points on the manifold."""
        ax.scatter(point[..., 0], point[..., 1], **kwargs)

    def plot_vector_field(self, ax, point, tangent_vec, **kwargs):
        """Quiver plot on the manifold."""
        point, tangent_vec = gs.to_ndarray(point, to_ndim=2), gs.to_ndarray(
            tangent_vec, to_ndim=2
        )
        n_points, n_vecs = point.shape[0], tangent_vec.shape[0]
        if n_points > 1 and n_vecs != n_points:
            raise ValueError("Each tangent vector must be associated to a base point.")

        ax.quiver(
            point[..., 0],
            point[..., 1],
            tangent_vec[..., 0],
            tangent_vec[..., 1],
            **kwargs
        )

    def plot_geodesic(
        self,
        ax,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
        times=gs.linspace(0, 1, 10),
        **kwargs
    ):
        """Plot geodesic on the manifold."""
        geodesic = self.space.metric.geodesic(
            initial_point, end_point, initial_tangent_vec
        )
        geod_at_t = geodesic(times)
        ax.plot(*gs.transpose(geod_at_t)[::-1], **kwargs)

    def plot_grid(
        self, ax, lower_left, upper_right, n_cells=[5, 5], steps=None, **kwargs
    ):
        """Plot geodesic grid on the manifold."""
        if n_cells is not None:
            n_cells = gs.array(n_cells) * gs.ones(2, dtype=int)
            h_points = gs.linspace(lower_left[0], upper_right[0], n_cells[0])
            v_points = gs.linspace(lower_left[1], upper_right[1], n_cells[1])
        else:
            if steps is not None:
                steps = gs.array(steps) * gs.ones(2)
                h_points = gs.arange(lower_left[0], upper_right[0], steps[0])
                v_points = gs.arange(lower_left[1], upper_right[1], steps[1])
            else:
                raise ValueError(
                    "Either the number of cells of the grid or the steps must input."
                )

        grid = gs.array(
            [[[h_point, v_point] for v_point in v_points] for h_point in h_points]
        )

        h_size, v_size = len(h_points), len(v_points)

        for i in range(h_size):
            for j in range(v_size):
                if i < h_size - 1:
                    self.plot_geodesic(
                        ax=ax,
                        initial_point=grid[i, j],
                        end_point=grid[i + 1, j],
                        **kwargs
                    )
                if j < v_size - 1:
                    self.plot_geodesic(
                        ax=ax,
                        initial_point=grid[i, j],
                        end_point=grid[i, j + 1],
                        **kwargs
                    )

    def plot_geodesic_ball(self, ax, center, n_rays=13, ray_norm=1, **kwargs):
        """Plot geodesic ball on the manifold."""
        theta = gs.linspace(0, 2 * gs.pi, n_rays)
        directions = gs.transpose(gs.stack((gs.cos(theta), gs.sin(theta))))
        tangent_vec = ray_norm * self.space.metric.normalize(
            directions, base_point=center
        )
        self.plot_geodesic(
            ax=ax, initial_point=center, initial_tangent_vec=tangent_vec, **kwargs
        )
