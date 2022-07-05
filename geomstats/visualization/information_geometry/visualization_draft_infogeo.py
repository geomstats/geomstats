"""Draft of visualizer (plotter) for information geometry."""

import io
from abc import ABC
from itertools import product

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

import geomstats.backend as gs

mpl.rc("figure", max_open_warning=0)


class Plotter:
    def __init__(self, figsize=(12, 8), **kwargs):
        self.figsize = figsize
        self.fig = plt.figure(figsize=self.figsize, **kwargs)

    def close():
        plt.close()


class Visualizer(ABC, Plotter):
    def __init__(self, fig, position, space, **kwargs):
        self.fig = fig
        self.position = position
        self.ax = self.fig.add_subplot(position, projection=self.projection)
        self.space = space
        self.dim = self.space.dim

    def scatter(self, point, **kwargs):
        """Plot points on the manifold.

        Parameters
        ----------
        ax : matplotlib window
            Location of the plot.
        point : array-like, shape=[...,dim]
            Point on the manifold.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        shape = point.shape
        n_points = shape[0]
        if 2 <= self.dim <= 3:
            self.ax.scatter(*gs.transpose(point), **kwargs)
        if self.dim == 1:
            self.ax.scatter(point, [0] * n_points, **kwargs)

    def plot_vector_field(self, base_point, tangent_vec, **kwargs):
        """Quiver plot on the manifold.

        Parameters
        ----------
        ax : matplotlib window
            Location of the plot.
        point : array-like, shape=[...,dim]
            Base point.
        tangent_vec : array-like, shape=[...,dim]
            Tangent vector at base point point.
        """
        base_point, tangent_vec = gs.to_ndarray(base_point, to_ndim=2), gs.to_ndarray(
            tangent_vec, to_ndim=2
        )
        n_points, n_vecs = base_point.shape[0], tangent_vec.shape[0]
        if n_points > 1 and n_vecs != n_points:
            raise ValueError("Each tangent vector must be associated to a base point.")

        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        if 2 <= self.dim <= 3:
            self.ax.quiver(
                *gs.transpose(base_point), *gs.transpose(tangent_vec), **kwargs
            )

    def plot_geodesic(
        self,
        initial_point,
        initial_tangent_vec=None,
        end_point=None,
        times=gs.linspace(0, 1, 100),
        **kwargs
    ):
        """Plot geodesic on the manifold.

        Parameters
        ----------
        ax : matplotlib window
            Location of the plot.
        initial_point : array-like, shape=[...,dim]
            Start point of the geodesic.
        end_point : array-like, shape=[...,dim]
            End point of the geodesic.
        initial_tangent_vec : array-like, shape=[...,dim]
            Tangent vector at base point initial_point.
        times : list
            Times at which to compute and plot the geodesic.
        """
        geodesic = self.space.metric.geodesic(
            initial_point, end_point, initial_tangent_vec
        )
        geod_at_t = geodesic(times)
        self.ax.plot(*gs.transpose(geod_at_t), **kwargs)

    def plot_grid(self, first_point, last_point, n_cells=5, steps=None, **kwargs):
        """Plot geodesic grid on the manifold.

        Parameters
        ----------
        ax : matplotlib window
            Location of the plot.
        first_point : array-like, shape=[dim,]
            First (i.e. with minimal coordinates) point of the geodesic grid.
        last_point : array-like, shape=[dim,]
            Last (i.e. with maximal coordinates) point of the geodesic grid.
        n_cells : float or list, shape=(,) or [dim,]
            Wanted number of cells in the grid, horizontal and vertical.
        steps : float or list, shape=(,) or [dim,]
            Wanted step size between points in the grid, horizontally and vertically.
        """
        if n_cells is not None:
            n_cells = gs.array(n_cells) * gs.ones(self.dim, dtype=int) + gs.ones(
                self.dim, dtype=int
            )
        else:
            if steps is not None:
                steps = gs.array(steps) * gs.ones(self.dim)
            else:
                raise ValueError(
                    "Either the number of cells of the grid or the steps must input."
                )

        grid = gs.array(
            [
                gs.linspace(first_point[i], last_point[i], n_cells[i])
                for i in range(self.dim)
            ]
        )
        meshgrid = gs.reshape(
            gs.array(list(product(*grid))),
            (
                *n_cells,
                self.dim,
            ),
        )

        combinations = list(product(*list([list(range(n_cell)) for n_cell in n_cells])))

        for combination in combinations:
            grid_point = meshgrid[combination]
            for i in range(self.dim):
                if combination[i] < n_cells[i] - 1:
                    next = gs.zeros(self.dim, int)
                    next[i] = 1
                    next_combination = tuple(gs.array(combination) + next)
                    next_point = meshgrid[next_combination]
                    self.plot_geodesic(
                        initial_point=grid_point, end_point=next_point, **kwargs
                    )

    def plot_geodesic_ball(self, center, n_rays=13, ray_norm=1, **kwargs):
        """Plot geodesic ball on the manifold.

        Parameters
        ----------
        ax : matplotlib window
            Location of the plot.
        center : array-like, shape=[...,dim]
            Center point of the geodesic ball.
        n_rays : int
            Wanted number of rays departing from the center.
        ray_norm : float
            Radius of the geodesic ball.
        """
        directions = self.directions(n_rays)
        tangent_vec = ray_norm * self.space.metric.normalize(
            directions, base_point=center
        )
        self.plot_geodesic(
            initial_point=center, initial_tangent_vec=tangent_vec, **kwargs
        )


class Visualizer1D(Visualizer):
    """Visualizer for 1-D parameter distributions."""

    def __init__(self, ax, space):
        super(Visualizer1D, self).__init__(ax=ax, space=space)

    # def scatter(self, ax, point, **kwargs):
    #     """Plot points on the manifold.

    #     Parameters
    #     ----------
    #     ax : matplotlib window
    #         Location of the plot.
    #     point : array-like, shape=[...,1]
    #         Point on the manifold.
    #     """
    #     n_points = len(point)
    #     ax.scatter(point, gs.zeros(n_points), **kwargs)

    def directions(self, n_rays):
        return gs.array([[-1.0], [1.0]])

    # def plot_geodesic_ball(self, ax, center, ray_norm=1, **kwargs):
    #     """Plot geodesic ball on the manifold.

    #     Parameters
    #     ----------
    #     ax : matplotlib window
    #         Location of the plot.
    #     center : array-like, shape=[...,1]
    #         Center of the geodesic ball.
    #     ray_norm : float
    #         Radius of the geodesic ball.
    #     """
    #     one = gs.ones(1)
    #     left = self.space.metric.exp(-ray_norm * one, center)
    #     right = self.space.metric.exp(ray_norm * one, center)
    #     ax.plot([left, right], gs.zeros(2), "bo", **kwargs)

    def iso_visualizer1D(self, ax, left_bound, right_bound, n_points, **kwargs):
        """Create framework for isometric representation of the manifold.

        Parameters
        ----------
        ax : matplotlib window
            Location of the plot.
        left_bound : float
            Left bound of the immersion (minimum point)
        right_bound : float
            Right bound of the immersion (maximum point)
        n_points : int
            Number of points to plot.
        """
        parent = self

        class Iso_Visualizer1D:
            """Isometric visualizer for 1D distributions."""

            def __init__(self, ax, left_bound, right_bound, n_points, **kwargs):
                self.space = parent.space
                self.ax = ax
                self.left_bound = left_bound
                self.right_bound = right_bound
                self.n_points = n_points

                def f(theta):
                    return self.space.metric.dist(theta, left_bound)

                df = gs.autodiff.value_and_grad(f)
                self.geod_distance = lambda theta: gs.sqrt(df(theta)[-1] - 1)

                # (thetas, geod_distance(thetas)) is the isometric curve on R^2, i.e.
                # the geodesic distance between theta1 and theta2 is exactly the
                # euclidean length of the curve between points at theta1 and theta2.

                times = gs.linspace(left_bound, right_bound, n_points)
                ax.plot(times, [self.geod_distance(time) for time in times], **kwargs)

            def isometric_scatter(self, point, **kwargs):
                """Plot points on the isometric manifold.

                Parameters
                ----------
                ax : matplotlib window
                    Location of the plot.
                point : array-like, shape=[...,1]
                    Point on the manifold.
                """
                self.ax.scatter(point, [self.geod_distance(pt) for pt in point])

            def isometric_plot_geodesic_ball(self, center, ray_norm=1, **kwargs):
                """Plot geodesic ball on the isometric manifold.

                Parameters
                ----------
                ax : matplotlib window
                    Location of the plot.
                center : array-like, shape=[...,1]
                    Center of the geodesic ball.
                ray_norm : float
                    Radius of the geodesic ball.
                """
                one = gs.ones(1)
                left = self.space.metric.exp(-ray_norm * one, center)
                right = self.space.metric.exp(ray_norm * one, center)
                n_points = int(
                    (right - left)
                    / (self.right_bound - self.left_bound)
                    * self.n_points
                )
                times = gs.linspace(left, right, n_points)
                self.ax.plot(
                    times, [self.geod_distance(time) for time in times], **kwargs
                )

        return Iso_Visualizer1D(ax, left_bound, right_bound, n_points, **kwargs)


class Visualizer2D(Visualizer):
    """Visualizer for 2-D parameter distributions."""

    def __init__(self, fig, position, space):
        self.projection = None
        super(Visualizer2D, self).__init__(fig=fig, position=position, space=space)

    def directions(self, n_rays=13):
        theta = gs.linspace(0, 2 * gs.pi, n_rays)
        return gs.transpose(gs.stack((gs.cos(theta), gs.sin(theta))))

    def overlay_scatter(self, ax, point, support, **kwargs):
        """Plot points on the manifold, p.d.f plot appears when overing over point.

        Parameters
        ----------
        ax : matplotlib window
            Location of the plot.
        point : array-like, shape=[...,2]
            Point on the manifold.
        support : list, shape=[2]
            Support of the p.d.f.
        """
        # define a function which returns an image as numpy array from figure
        def get_img_from_fig(fig, dpi=180):
            """Get image from matplotlib plot."""
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        left_bound, right_bound = support
        x = gs.linspace(left_bound, right_bound, 100)
        fig_template = plt.figure()
        plt.plot(x, self.space.point_to_pdf(gs.array([1, 1]))(x))

        template_plot = get_img_from_fig(fig_template)

        # create figure and plot scatter
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        (line,) = ax.plot(point[:, 0], point[:, 1], ls="", marker="o")

        # create the annotations box
        im = OffsetImage(template_plot, zoom=1)
        xybox = (240.0, 160.0)
        ab = AnnotationBbox(
            im,
            (0, 0),
            xybox=xybox,
            xycoords="data",
            boxcoords="offset points",
            pad=0.3,
            arrowprops=dict(arrowstyle="->"),
        )
        # add it to the axes and make it invisible
        ax.add_artist(ab)
        ab.set_visible(False)

        def hover(event):
            # if the mouse is over the scatter points
            if line.contains(event)[0]:
                # find out the index within the array from the event
                (ind,) = line.contains(event)[1]["ind"]
                # get the figure size
                w, h = fig.get_size_inches() * fig.dpi
                ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
                hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)
                # if event occurs in the top or right quadrant of the figure,
                # change the annotation box position relative to mouse.
                ab.xybox = (xybox[0] * ws, xybox[1] * hs)
                # make annotation box visible
                ab.set_visible(True)
                # place it at the position of the hovered scatter point
                ab.xy = (point[ind, 0], point[ind, 1])
                # draw the plot
                fig_plot = plt.figure(dpi=fig.dpi)
                ax = fig_plot.add_subplot(111)
                ax.plot(x, self.space.point_to_pdf(point[ind])(x))
                # set the image corresponding to that point
                array = get_img_from_fig(fig_plot)
                img = Image.fromarray(array, "RGB")
                img = img.resize((240, 160))
                img.save("my.png")
                im.set_data(plt.imread("my.png"))
            else:
                # if the mouse is not over a scatter point
                ab.set_visible(False)
            fig.canvas.draw_idle()

        # add callback for mouse moves
        fig.canvas.mpl_connect("motion_notify_event", hover)
