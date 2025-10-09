import numpy as np
from geomstats.information_geometry.categorical import (
    CategoricalDistributions, CategoricalMetric)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CategoricalDistributionsManifold:
    r"""Class for visualizing the manifold of categorical distributions.

    This is the set of $n+1$-tuples of positive reals that sum up to one,
    i.e. the $n$-simplex. Each point is the parameter of a categorical
    distribution, i.e. gives the probabilities of $n$ different outcomes
    in a single experiment.

    Attributes:
    -----------
    dim : integer
        Dimension of the manifold.
    points: array-like, [[..., dim + 1], [..., dim + 1], ... ]
        Discrete points to be plotted on the manifold.

    Notes:
    ------
    The class only implements visualization methods for 2D and 3D manifolds.
    """

    def __init__(self, dim):
        """Construct a CategoricalDistributionsManifold object.

        Construct a CategoricalDistributionsManifold with a given dimension.

        Parameters:
        -----------
        dim : integer
            Dimension of the manifold

        Returns:
        --------
        None.

        Notes:
        ------
        dim should be a positive integer.
        The methods only support visualization of 2-D and 3-D manifolds.
        """
        self.dim = dim
        self.points = []
        self.ax = None
        self.elev, self.azim = None, None
        self.metric = CategoricalMetric(dim=self.dim)
        self.dist = CategoricalDistributions(dim=self.dim)

    def plot(self):
        """Plot the 2D or 3D Manifold.

        Plot the 2D Manifold as a regular 2-simplex(triangle) or
        the 3D Manifold as a regular 3-simplex(tetrahedral).

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        Notes
        -----
        This method only works properly if the dimension is 2 or 3.

        References
        ----------
        Simplex: https://en.wikipedia.org/wiki/Simplex
        """
        min_limit = 0
        max_limit = 1
        plt.figure(dpi=100)
        self.set_axis(min_limit, max_limit)
        if self.dim == 3:
            self.set_view()
            x = [0, 1, 0, 0]
            y = [0, 0, 1, 0]
            z = [0, 0, 0, 1]
            vertices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            tupleList = list(zip(x, y, z))
            poly3d = [
                [tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))]
                for ix in range(len(vertices))
            ]
            self.ax.add_collection3d(
                Poly3DCollection(
                    poly3d,
                    edgecolors="k",
                    facecolors=(0.9, 0.9, 0.9, 1.0),
                    linewidths=3,
                    alpha=0.2,
                )
            )

        elif self.dim == 2:
            X = np.linspace(start=min_limit, stop=max_limit, num=101,
                            endpoint=True)
            Y = 1 - X
            self.ax.fill_between(X, Y, color=(0.9, 0.9, 0.9, 1.0))
            self.ax.set_title("2 Dimension Categorical Manifold")

    def set_points(self, points):
        self.points = points

    def scatter(self, n_samples, **scatter_kwargs):
        """Scatter plot some randomly sampled points in the manifold.

        Plot the manifold along with some randomly sampled points
        lying on the manifold.

        Parameters:
        -----------
        n_samples : integer
            The number of randomly sampled points.

        **scatter_kwargs: optional
            Inherits the matplotlib scatter function parameters.

        Returns:
        --------
        None.

        Notes:
        ------
        This method internally calls the plot method.
        """
        self.set_points(self.dist.random_point(n_samples=n_samples))
        self.plot()
        if self.dim == 2:

            for point in self.points:
                self.ax.scatter(point[0], point[1], **scatter_kwargs)
            self.ax.set_title(
                f"2 Dimension Categorical Manifold with {n_samples} Samples"
            )
        elif self.dim == 3:
            for point in self.points:
                self.ax.scatter(point[0], point[1], point[2], **scatter_kwargs)
            self.ax.set_title(
                f"3 Dimension Categorical Manifold with {n_samples} Samples"
            )

        self.clear_points()

    def plot_geodesic(self, initial_point, end_point=None,
                      tangent_vector=None):
        """Plot a geodesic on the manifold.

        Plot a geodesic that is either specified with
        1) an initial_point and an end_point, or
        2) an initial point and an initial tangent vector
        on the manifold.

        Parameters:
        -----------
        initial_point: array-like, shape = [..., dim + 1]
            Initial point on the manifold.

        end_point: optional, array-like, shape = [..., dim + 1]
            End point on the manifold.

        tangent_vector: optional, array-like, shape = [..., dim + 1]
            Initial tangent vector at the initial point.

        Returns:
        --------
        None.

        Notes:
        ------
        Either end_point or tangent_vector needs to be specified.
        The initial point will be marked red.
        The initial tangent vector will also be plotted starting from
        the initial point.
        """
        self.plot()
        geodesic = self.metric.geodesic(
            initial_point=initial_point,
            end_point=end_point,
            initial_tangent_vec=tangent_vector,
        )
        num_samples = 200
        if self.dim == 2:
            for i in range(num_samples):
                point = geodesic(i / num_samples)
                self.ax.scatter(point[0], point[1], color="blue", s=2)
                self.ax.scatter(geodesic(0)[0], geodesic(0)[1],
                                color="red", s=30)
            if tangent_vector is not None:
                normalized_tangent_vector = tangent_vector / np.sum(
                    np.power(tangent_vector, 2)
                )
                self.ax.quiver(
                    initial_point[0],
                    initial_point[1],
                    normalized_tangent_vector[0],
                    normalized_tangent_vector[1],
                    color="red",
                    angles="xy",
                    scale_units="xy",
                    scale=10,
                )
        else:
            for i in range(num_samples):
                point = geodesic(i / num_samples)
                self.ax.scatter(point[0], point[1], point[2], color="blue",
                                s=2)
                self.ax.scatter(
                    geodesic(0)[0], geodesic(0)[1], geodesic(0)[2],
                    color="red", s=30
                )
            if tangent_vector is not None:
                normalized_tangent_vector = tangent_vector / (
                    np.sum(np.power(tangent_vector, 2)) * 3
                )
                self.ax.quiver(
                    initial_point[0],
                    initial_point[1],
                    initial_point[2],
                    normalized_tangent_vector[0],
                    normalized_tangent_vector[1],
                    normalized_tangent_vector[2],
                    color="red",
                    # angles = 'xy',
                    # scale_units = 'xy',
                    # scale = 10,
                )

    def plot_log(self, end_point, base_point):
        """Plot the result of taking the logarithm of two points on the manifold.

        Plot the tangent vector calculated from taking the logarithm between
        the two input points on the manifold.

        Parameters:
        -----------
        end_point: array-like, shape = [..., dim + 1]
            End point on the manifold.

        base_point: array-like, shape = [..., dim + 1]
            Base point on the manifold.

        Returns:
        --------
        None.
        """
        tangent_vec = self.metric.log(point=end_point, base_point=base_point)
        self.plot_helper(
            end_point=end_point,
            base_point=base_point,
            tangent_vec=tangent_vec,
            operation="Log",
        )

    def plot_exp(self, tangent_vec, base_point):
        """Plot the result of taking the exponential of one point with
        one of its tangent vector.

        Plot the end point resulting from taking the exponential of the base
        point with respect to a tangent vector.

        Parameters:
        -----------
        tangent_vec: array-like, shape = [..., dim + 1]
            A tangent vector at the base point.

        base_point: array-like, shape = [..., dim + 1]
            Base point on the manifold.

        Returns:
        --------
        None.
        """
        end_point = self.metric.exp(tangent_vec=tangent_vec,
                                    base_point=base_point)
        self.plot_helper(
            end_point=end_point,
            base_point=base_point,
            tangent_vec=tangent_vec,
            operation="Exp",
        )

    def plot_helper(self, end_point, base_point, tangent_vec, operation):
        """Plot two points and a vector together on a manifold.

        Helper function used by plot_exp and plot_log methods.

        Parameters:
        -----------
        end_point: array-like, shape = [..., dim + 1]
            End point on the manifold.

        base_point: array-like, shape = [..., dim + 1]
            Base point on the manifold.

        tangent_vec: array-like, shape = [..., dim + 1]
            A tangent vector to the manifold.

        Returns:
        --------
        None.

        Notes:
        ------
        The base point and the tangent vector will be marked red.
        THe end point will be marked blue.

        """
        self.plot()
        self.ax.set_title(
            f"{operation} Operation with {self.dim} Dimension Categorical\
            Manifold"
        )
        if self.dim == 3:
            # Plot in Matplotlib
            self.ax.scatter(
                base_point[0], base_point[1], base_point[2], color="red", s=30
            )
            self.ax.scatter(
                end_point[0], end_point[1], end_point[2], color="blue", s=30
            )
            self.ax.quiver(
                base_point[0],
                base_point[1],
                base_point[2],
                tangent_vec[0],
                tangent_vec[1],
                tangent_vec[2],
                color="red",
                length=0.1,
                normalize=True,
            )

        if self.dim == 2:
            self.ax.scatter(base_point[0], base_point[1], color="red", s=30)
            self.ax.scatter(end_point[0], end_point[1], color="blue", s=30)
            self.ax.quiver(
                base_point[0],
                base_point[1],
                tangent_vec[0],
                tangent_vec[1],
                color="red",
                angles="xy",
                scale_units="xy",
                scale=5,
            )

    def plot_grid(self):
        """Plot the manifold with a geodesic grid.

        Plot some geodesic grid lines on top of a 2D manifold.

        Parameters:
        -----------
        None.

        Returns:
        --------
        None.

        Notes:
        ------
        This function only works for 2D manifold.
        """
        self.plot()
        points = [
            np.array([0.5, 0, 0.5]),
            np.array([0, 0.5, 0.5]),
            np.array([0.5, 0.5, 0]),
            np.array([0.25, 0, 0.75]),
            np.array([0, 0.25, 0.75]),
            np.array([0.75, 0, 0.25]),
            np.array([0, 0.75, 0.25]),
        ]

        num_samples = 100
        curves = [
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 2),
            (4, 2),
            (3, 4),
            (5, 2),
            (6, 2),
            (5, 6),
        ]
        for curve in curves:
            geodesic = self.metric.geodesic(
                initial_point=points[curve[0]], end_point=points[curve[1]]
            )
            for i in range(num_samples):
                point = geodesic(i / num_samples)
                self.ax.scatter(point[0], point[1], color="black", s=1)

    def clear_points(self):
        """Clear the points stored in the object.

        Clear the points vector stored as an attribute of an object.

        Parameters:
        -----------
        None.

        Returns:
        --------
        None.
        """
        self.points = []

    def set_axis(self, min_limit, max_limit):
        """Set the axes in 2D or 3D Euclidean space.

        Set the boundary of each axis for plotting
        as specified in the input for plotting the manifold.

        Parameters:
        -----------
        min_limit : float
            Lower limit for each axis.
        min_limit : float
            Upper limit for each axis.

        Returns:
        --------
        None.

        Notes:
        ------
        This method is not intended to be called externally.
        """
        if self.dim == 3:
            ax = plt.subplot(111, projection="3d")
            plt.setp(
                ax,
                xlim=(min_limit, max_limit),
                ylim=(min_limit, max_limit),
                zlim=(min_limit, max_limit),
                anchor=(0, 0),
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
            )

        elif self.dim == 2:
            ax = plt.subplot(111)
            plt.setp(
                ax,
                xlim=(min_limit, max_limit),
                ylim=(min_limit, max_limit),
                xlabel="X",
                ylabel="Y",
                aspect="equal",
            )

        self.ax = ax

    def set_view(self, elev=30.0, azim=20.0):
        """Set the viewing angle for plotting the 3D manifold.

        Set the elevation and azimuthal angle of viewing the 3D manifold.

        Parameters
        ----------
        elev : float
            Angle of elevation from the x-y plane in degrees (default: 30.0).
        azim : float
            Azimuthal angle in the x-y plane in degrees (default: 20.0).

        Returns
        -------
        None.
        """
        if self.dim == 3:
            if self.ax is None:
                self.set_axis()
            self.elev, self.azim = elev, azim
            self.ax.view_init(elev, azim)
