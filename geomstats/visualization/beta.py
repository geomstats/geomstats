"""Visualization for Geometric Statistics."""
import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.information_geometry.beta import BetaDistributions

SE2_GROUP = SpecialEuclidean(n=2, point_type="matrix")

beta = BetaDistributions()


class Beta:
    """Visualization for Beta Distributions"""

    def examine_points(self, points, **kwargs):
        """Examine all input manifold points.

        Confirms that points passed into function lie on manifold.
        Prepares points for plotting.

        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Beta manifold points to be plotted.
        """
        points = gs.array(points)
        if len(points.shape) == 1:
            points = gs.expand_dims(points, axis=0)

        if not len(points) > 0:
            raise ValueError("No points given")
        if not gs.all(points > 0):
            raise ValueError(
                "Points must be in the upper-right quadrant of Euclidean space"
            )
        if not ((points.shape[-1] == 2 and len(points.shape) == 2)):
            raise ValueError("Points must lie in 2D space")
        limit = gs.amax(points)
        limit += limit / 10
        return points, limit

    def plot(self, points, size=None, **kwargs):
        """Draw the beta manifold points in the parameter space.
        
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Beta manifold points to be plotted.
        size : array-like, shape=[..., 2]
            Defines the range of the manifold to be shown.
            Optional, default: None
        """

        points, limit = self.examine_points(points)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        if not size:
            ax.set(xlim=(0, limit), ylim=(0, limit))
        else:
            ax.set(xlim=(0, size[0]), ylim=(0, size[1]))
        ax.scatter(points[:, 0], points[:, 1], **kwargs)
        plt.title("Points on 2D manifold of beta distributions")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")

    def plot_rendering(self, initial_point=[2, 2], size=[10, 10], sampling_period=1):
        """Draw grid points in a given range.
        
        Parameters
        ----------
        Initial_point : array-like, shape=[1, 2]
            Defines initial point for plot rendering
            Optional, default: [2,2]
        size : array-like, shape=[..., 2]
            Defines the range of the samples to be shown
            Optional, default: [10,10]
        sampling_period : float, >0
            Defines the sampling period of the sampled data
            Optional, default: 1
        """

        for value in initial_point:
            if value < 0:
                raise ValueError(
                    "Initial Point {} is not in the first quadrant".format(
                        initial_point
                    )
                )

        sz = gs.array(size)
        if sz.size != 2:
            raise ValueError("size should be a 1*2 array")
        x = gs.linspace(
            initial_point[0],
            (initial_point[0] + size[0] - 1) * sampling_period,
            size[0],
        )
        y = gs.linspace(
            initial_point[1],
            (initial_point[1] + size[1] - 1) * sampling_period,
            size[1],
        )
        points = [[i, j] for i in x for j in y]
        points_x = [i[0] for i in points]
        points_y = [i[1] for i in points]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        ax.scatter(points_x, points_y)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")

    def plot_grid(
        self, size, initial_point=[2, 2], n_steps=100, n_points=10, step=1, **kwargs
    ):
        """Draw the grids on beta manifold.
        
        Parameters
        ----------
        size : array-like, shape=[..., 2]
            Defines the range of the grids to be shown.
        initial_point : array-like, shape=[1,2]
            Defines the initial point for plotting the beta manifold grid.
            Optional, default: [2,2]
        n_steps : int, >0
            Defines the number of steps for integration.
            Optional, default: 100
        n_points : int, >0
            Defines the number of points for interpolation.
            Optional, default: 10
        step : float, >0
            Defines the length of a step for the grid
            Optional, default: 1
        """

        for value in initial_point:
            if value < 0:
                raise ValueError(
                    "Initial Point {} is not in the first quadrant".format(
                        initial_point
                    )
                )

        sz = gs.array(size)
        if sz.size != 2:
            raise ValueError("size should be a 1*2 array")
        xsteps = [(initial_point[0] + i * step) for i in range(size[0])]
        ysteps = [(initial_point[1] + i * step) for i in range(size[1])]

        t = gs.linspace(0, 1, n_points)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        for i in xsteps:
            for j in ysteps:
                start = [i, j]
                end_h = [i + step, j]
                end_v = [i, j + step]
                if i < (size[0] + initial_point[0] - 1):
                    grid_h = beta.metric.geodesic(
                        initial_point=start, end_point=end_h, n_steps=n_steps
                    )
                    ax.plot(*gs.transpose(gs.array([grid_h(k) for k in t])))
                if j < (size[1] + initial_point[1] - 1):
                    grid_v = beta.metric.geodesic(
                        initial_point=start, end_point=end_v, n_steps=n_steps
                    )
                    ax.plot(*gs.transpose(gs.array([grid_v(k) for k in t])))
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")
        plt.title("Grids in the beta manifold")

    def scatter(self, points, **kwargs):
        """Draw the scatter plot of given beta manifold points.
        
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Manifold point representing a beta distribution.  
        """

        points, limit = self.examine_points(points)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.set(xlim=(0, limit), ylim=(0, limit))
        ax.scatter(points[:, 0], points[:, 1], **kwargs)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")
        ax.set_title("Scatter plot of beta manifolds")

    def plot_geodesic(
        self,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
        n_steps=100,
        n_points=10,
        **kwargs,
    ):
        """Draw the geodesic line from a single beta manifold point.
        
        Parameters
        ----------
        initial_point : array-like, shape=[1, 2]
            Starting point representing a beta distribution.
        end_point : array-like, shape=[1, 2]
            Ending point representing a beta distribution.
            Optional, default: None.
        initial_tangent_vec : array-like, shape=[1, 2]
            Initial tangent vector for the starting point.
            Optional, default: None.
        n_steps : int, >0
            Number of steps for integration.
            Optional, default: 100.
        n_points : int, >0
            Number of points for interpolation.
            Optional, default: 10.
        """

        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )

        t = gs.linspace(0, 1, n_points)

        if end_point is not None:

            for point in [initial_point, end_point]:
                x, y = point
                if x < 0 or y < 0:
                    raise ValueError(
                        "Point {} is not in the first quadrant".format(point)
                    )

            u_lim = gs.amax(gs.array([initial_point, end_point])) + 1
            l_lim = gs.amin(gs.array([initial_point, end_point])) - 1
            geod = beta.metric.geodesic(
                initial_point=initial_point, end_point=end_point, n_steps=n_steps
            )(t)

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.set(xlim=(l_lim, u_lim), ylim=(l_lim, u_lim))
            ax.scatter(geod[:, 0], geod[:, 1], **kwargs)
            ax.set_title(
                "Geodesic between two beta distributions for Fisher-Rao metric"
            )
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\beta$")

        if initial_tangent_vec is not None:
            if (initial_point < 0).any():
                raise ValueError("Initial point is not in the first quadrant")
            geod = beta.metric.geodesic(
                initial_point=initial_point,
                initial_tangent_vec=initial_tangent_vec,
                n_steps=n_steps,
            )(t)
            u_lim = gs.amax(geod) + 1
            l_lim = gs.amin(geod) - 1
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.set(xlim=(l_lim, u_lim), ylim=(l_lim, u_lim))
            ax.scatter(geod[:, 0], geod[:, 1], **kwargs)
            ax.set_title(
                "Geodesic between two beta distributions for Fisher-Rao metric"
            )
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\beta$")

    def get_vector_field(
        self,
        initial_point,
        tangent_vecs,
        ray_length=10,
        n_points=50,
        n_steps=10,
        **kwargs,
    ):
        """Calculate vector field given initial point and tangent vectors.
        
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Point representing a beta distribution.
        """

        if not gs.all(initial_point > 0):
            raise ValueError(
                "Points must be in the upper-right quadrant of Euclidean space"
            )
        if not (initial_point.shape[-1] == 2 and len(initial_point.shape) == 2):
            raise ValueError("Points must lie in 2D space")
        if len(tangent_vecs.shape) != 2 or tangent_vecs.shape[1] != 2:
            raise ValueError("Tangent vector needs to be of shape N x 2")
        scaled_tangent_vecs = ray_length * tangent_vecs

        t = gs.linspace(0, 1, n_points)
        for j in range(len(scaled_tangent_vecs)):
            geod = beta.metric.geodesic(
                initial_point=initial_point,
                initial_tangent_vec=scaled_tangent_vecs[j, :],
                n_steps=n_steps,
            )
            geod = gs.transpose(gs.array([geod(k) for k in t]))
            geod = gs.expand_dims(geod, 0)
            if j == 0:
                geods = geod
            else:
                geods = gs.vstack((geods, geod))
        x_lower_limit = gs.amin(geods[:, 0, :])
        x_lower_limit -= x_lower_limit / 10
        x_upper_limit = gs.amax(geods[:, 0, :])
        x_upper_limit += x_upper_limit / 10
        y_lower_limit = gs.amin(geods[:, 1, :])
        y_lower_limit -= y_lower_limit / 10
        y_upper_limit = gs.amax(geods[:, 1, :])
        y_upper_limit += y_upper_limit / 10
        xlims = [x_lower_limit, x_upper_limit]
        ylims = [y_lower_limit, y_upper_limit]

        return geods, xlims, ylims

    def plot_vector_field(
        self,
        initial_point,
        tangent_vecs,
        ray_length=0.25,
        n_points=50,
        n_steps=10,
        **kwargs,
    ):
        """Draw the vector field of the beta manifold.
        
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Point representing a beta distribution.
        """

        center, _ = self.examine_points(initial_point)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        geods, xlims, ylims = self.get_vector_field(
            center, tangent_vecs, ray_length, n_points, n_steps
        )

        for geod in geods:
            ax.plot(*geod)
        ax.scatter(center[:, 0], center[:, 1])
        ax.set(xlim=(xlims[0], xlims[1]), ylim=(ylims[0], ylims[1]))
        ax.set_title("Vector field in the manifold of beta distributions")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")

    def plot_geodesic_ball(
        self,
        initial_point,
        n_rays,
        ray_length,
        n_steps=100,
        n_points=10,
        **kwargs,
    ):
        """Draw the geodesic ball of a point on the beta manifold.
        
        Parameters
        ----------
        inital_point : array-like, shape=[1, 2]
            Point representing a beta distribution.
        tangent_vecs : array-like, shape=[..., 2]
            Set of tangent vectors for geodesic ball.
        n_steps : int, >0
            Number of steps for integration.
            Optional, default: 100.
        n_points : int, >0
            Number of points for interpolation.
            Optional, default: 10.
        """

        center, _ = self.examine_points(initial_point)
        theta = gs.linspace(-gs.pi, gs.pi, n_rays)
        directions = gs.transpose(gs.stack((gs.cos(theta), gs.sin(theta))))
        direction_norms = beta.metric.squared_norm(directions, center)
        direction_norms = direction_norms ** (1 / 2)
        unit_vectors = directions / gs.expand_dims(direction_norms, 1)
        tangent_vecs = ray_length * unit_vectors

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        geods, xlims, ylims = self.get_vector_field(
            center, tangent_vecs, ray_length, n_points, n_steps
        )
        for geod in geods:
            ax.plot(*geod)
        ax.scatter(center[:, 0], center[:, 1])

        ax.set(xlim=(xlims[0], xlims[1]), ylim=(ylims[0], ylims[1]))
        ax.set_title("Geodesic ball in the manifold of beta distributions")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")
