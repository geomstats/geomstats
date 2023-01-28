r"""
Manifold of linear subspaces.

The Grassmannian :math:`Gr(n, k)` is the manifold of k-dimensional
subspaces in n-dimensional Euclidean space.

Lead author: Olivier Peltre.

:math:`Gr(n, k)` is represented by
:math:`n \times n` matrices
of rank :math:`k`  satisfying :math:`P^2 = P` and :math:`P^T = P`.
Each :math:`P \in Gr(n, k)` is identified with the unique
orthogonal projector onto :math:`{\rm Im}(P)`.

:math:`Gr(n, k)` is a homogoneous space, quotient of the special orthogonal
group by the subgroup of rotations stabilising a k-dimensional subspace:

.. math::

    Gr(n, k) \simeq \frac {SO(n)} {SO(k) \times SO(n-k)}

It is therefore customary to represent the Grassmannian
by equivalence classes of orthogonal :math:`k`-frames in :math:`{\mathbb R}^n`.
For such a representation, work in the Stiefel manifold instead.

.. math::

    Gr(n, k) \simeq St(n, k) / SO(k)

References
----------
[Batzies15]_    Batzies, E., K. Hüper, L. Machado, and F. Silva Leite.
                “Geometric Mean and Geodesic Regression on Grassmannians.”
                Linear Algebra and Its Applications 466 (February 1, 2015):
                83–101. https://doi.org/10.1016/j.laa.2014.10.003.
"""


import geomstats.backend as gs
import geomstats.errors
import matplotlib.pyplot as plt
import numpy as np
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


def projection_to_two_d(projections, points):

    """
    Takes Grassmannian generated projection matrix and converts
    it to a plotable 2D point.

    INPUT: Array of 2x2 projection matricies
    or a single projection matrix

    OUTPUT: Points on manifold but in a Euclidean representation

    """

    storage = []

    if points == 1:
        x = np.sqrt((projections[0][0]))
        y = (projections[0][1]) / x
        vector = [x, y]
        return vector
    for i in range(points):
        x = np.sqrt((projections[i][0][0]))
        y = (projections[i][0][1]) / x
        vector = [x, y]
        storage.append(vector)
    return np.asarray(storage)


def projection_to_three_d(projections, points):

    """
    Takes Grassmannian generated projection matrix and
    converts it to a plotable 3D point.

    INPUT: Array of 3x3 projection matricies or a single projection matrix

    OUTPUT: Points on manifold but in a Euclidean representation

    """

    storage = []

    vector = np.random.rand(3, 1)

    if points == 1:
        vector = np.matmul(projections, vector) / np.linalg.norm(
            np.matmul(projections, vector)
        )

        return vector
    for i in range(points):
        x, y, z = np.matmul(projections[i], vector) / np.linalg.norm(
            np.matmul(projections[i], vector)
        )
        vector = [x, y, z]
        storage.append(vector)
    return np.asarray(storage)


def two_d_to_projection(vector):

    """Takes 2D point on manifold and gets its corresponding projection.

    INPUT: Point on manifold as a 2x1 vector

    OUTPUT: Projection matrix for point

    WOULD BE USED FOR TANGENT REPRESENTATIONS

    """

    x = vector[0]
    theta = np.arccos(x)
    projection = np.zeros((2, 2))
    projection[0][0] = (np.cos(theta)) ** 2
    projection[0][1] = np.cos(theta) * np.sin(theta)
    projection[1][0] = projection[0][1]
    projection[1][1] = (np.sin(theta)) ** 2
    return projection


def submersion(point, k):
    r"""Submersion that defines the Grassmann manifold.

    The Grassmann manifold is defined here as embedded in the set of
    symmetric matrices, as the pre-image of the function defined around the
    projector on the space spanned by the first k columns of the identity
    matrix by (see Exercise E.25 in [Pau07]_).
    .. math:

            \begin{pmatrix} I_k + A & B^T \\ B & D \end{pmatrix} \mapsto
                (D - B(I_k + A)^{-1}B^T, A + A^2 + B^TB

    This map is a submersion and its zero space is the set of orthogonal
    rank-k projectors.

    References
    ----------
    .. [Pau07]   Paulin, Frédéric. “Géométrie diﬀérentielle élémentaire,” 2007.
                 https://www.imo.universite-paris-saclay.fr/~paulin
                 /notescours/cours_geodiff.pdf.
    """
    _, eigvecs = gs.linalg.eigh(point)
    eigvecs = gs.flip(eigvecs, -1)
    flipped_point = Matrices.mul(Matrices.transpose(eigvecs), point, eigvecs)
    b = flipped_point[..., k:, :k]
    d = flipped_point[..., k:, k:]
    a = flipped_point[..., :k, :k] - gs.eye(k)
    first = d - Matrices.mul(
        b, GeneralLinear.inverse(a + gs.eye(k)), Matrices.transpose(b)
    )
    second = a + Matrices.mul(a, a) + Matrices.mul(Matrices.transpose(b), b)
    row_1 = gs.concatenate([first, gs.zeros_like(b)], axis=-1)
    m_T = [Matrices.transpose(gs.zeros_like(b)), second]
    row_2 = gs.concatenate(m_T, axis=-1)
    return gs.concatenate([row_1, row_2], axis=-2)


def _squared_d_g_a(point_a, point_b, metric):
    """Compute gradient of squared_dist wrt point_a.

    Compute the Riemannian gradient of the squared geodesic
    distance with respect to the first point point_a.

    Parameters
    ----------
    point_a : array-like, shape=[..., dim]
        Point.
    point_b : array-like, shape=[..., dim]
        Point.
    metric : SpecialEuclideanMatrixCannonicalLeftMetric
        Metric defining the distance.

    Returns
    -------
    _ : array-like, shape=[..., dim]
        Riemannian gradient, in the form of a tangent
        vector at base point : point_a.
    """
    return -2 * metric.log(point_b, point_a)


def _squared_dist_grad_point_b(point_a, point_b, metric):
    """Compute gradient of squared_dist wrt point_b.

    Compute the Riemannian gradient of the squared geodesic
    distance with respect to the second point point_b.

    Parameters
    ----------
    point_a : array-like, shape=[..., dim]
        Point.
    point_b : array-like, shape=[..., dim]
        Point.
    metric : SpecialEuclideanMatrixCannonicalLeftMetric
        Metric defining the distance.

    Returns
    -------
    _ : array-like, shape=[..., dim]
        Riemannian gradient, in the form of a tangent
        vector at base point : point_b.
    """
    return -2 * metric.log(point_a, point_b)


@gs.autodiff.custom_gradient(_squared_d_g_a, _squared_dist_grad_point_b)
def _squared_dist(point_a, point_b, metric):
    """Compute geodesic distance between two points.

    Compute the squared geodesic distance between point_a
    and point_b, as defined by the metric.

    This is an auxiliary private function that:
    - is called by the method `squared_dist` of the class
    SpecialEuclideanMatrixCannonicalLeftMetric,
    - has been created to support the implementation
    of custom_gradient in tensorflow backend.

    Parameters
    ----------
    point_a : array-like, shape=[..., dim]
        Point.
    point_b : array-like, shape=[..., dim]
        Point.
    metric : SpecialEuclideanMatrixCannonicalLeftMetric
        Metric defining the distance.

    Returns
    -------
    _ : array-like, shape=[...,]
        Geodesic distance between point_a and point_b.
    """
    return metric.private_squared_dist(point_a, point_b)


class Grassmannian:
    """Class for Grassmann manifolds Gr(n, k).

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    k : int
        Dimension of the subspaces.
    """

    def __init__(self, n, k):
        geomstats.errors.check_integer(k, "k")
        geomstats.errors.check_integer(n, "n")
        if k > n:
            raise ValueError(
                "k <= n is required: k-dimensional subspaces in n dimensions."
            )

        self.n = n
        self.k = k

        self.dim = int(k * (n - k))
        self.embedding_space = SymmetricMatrices(n)
        # self.submersion = lambda x: submersion(x, k), value=gs.zeros((n, n))
        #         self.tangent_submersion = lambda v, x: 2 \
        #             * Matrices.to_symmetric(Matrices.mul(x, v)) \
        #             - v
        self.metric = GrassmannianCanonicalMetric(n, k)

    def random_uniform(self, n_samples=1):
        """Sample random points from a uniform distribution.

        Following [Chikuse03]_, :math: `n_samples * n * k` scalars are sampled
        from a standard normal distribution and reshaped to matrices,
        the projectors on their first k columns follow a uniform distribution.

        Parameters
        ----------
        n_samples : int
            The number of points to sample
            Optional. default: 1.

        Returns
        -------
        projectors : array-like, shape=[..., n, n]
            Points following a uniform distribution.

        References
        ----------
        .. [Chikuse03] Yasuko Chikuse, Statistics on special manifolds,
        New York: Springer-Verlag. 2003, 10.1007/978-0-387-21540-2
        """
        points = gs.random.normal(size=(n_samples, self.n, self.k))
        full_rank = Matrices.mul(Matrices.transpose(points), points)
        projector = Matrices.mul(
            points, GeneralLinear.inverse(full_rank), Matrices.transpose(points)
        )
        return projector[0] if n_samples == 1 else projector

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points from a uniform distribution.

        Following [Chikuse03]_, :math: `n_samples * n * k` scalars are sampled
        from a standard normal distribution and reshaped to matrices,
        the projectors on their first k columns follow a uniform distribution.

        Parameters
        ----------
        n_samples : int
            The number of points to sample
            Optional. default: 1.

        Returns
        -------
        projectors : array-like, shape=[..., n, n]
            Points following a uniform distribution.

        References
        ----------
        .. [Chikuse03] Yasuko Chikuse, Statistics on special manifolds,
        New York: Springer-Verlag. 2003, 10.1007/978-0-387-21540-2
        """
        return self.random_uniform(n_samples)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Compute the bracket (commutator) of the base_point with
        the skew-symmetric part of vector.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Vector.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        sym = Matrices.to_symmetric(vector)
        return Matrices.bracket(base_point, Matrices.bracket(base_point, sym))

    def projection(self, point):
        """Project a matrix to the Grassmann manifold.

        An eigenvalue decomposition of (the symmetric part of) point is used.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Projected point.
        """
        mat = Matrices.to_symmetric(point)
        _, eigvecs = gs.linalg.eigh(mat)
        diagonal = gs.array([0.0] * (self.n - self.k) + [1.0] * self.k)
        p_d = gs.einsum("...ij,...j->...ij", eigvecs, diagonal)
        return Matrices.mul(p_d, Matrices.transpose(eigvecs))

    def plot(self, ticks):
        """
        Plots a Grassmannian manifold using the projection matricies
        generated by the Grassmannian class.
        Works for cases with k=1 and n=2,3.
        Code will be generalized for future use.

        INPUT: Grassmannian manifold and True/False for ticks to determine
        if you want labels

        OUTPUT: None, plots manifold using MatPlotLib

        """

        n = self.n  # Size of space
        k = self.k  # Subspace size
        if k > 1 or n > 3:
            print(
                "Cannot handle sizes of k larger than 1 at this time.\
                Cannot handle n larger than 3 at this time."
            )
            return -1
        if n == 2:
            points = 10000
            projections = self.random_uniform(points)
            array_of_2d_vectors = projection_to_two_d(projections, points)
            for i in range(np.shape(array_of_2d_vectors)[0]):
                plt.plot(array_of_2d_vectors[i][0], array_of_2d_vectors[i][1], "r.")
            if ticks is True:
                pi = np.pi
                thetas = [
                    pi / 2,
                    pi / 3,
                    pi / 4,
                    pi / 6,
                    0,
                    11 * pi / 6,
                    7 * pi / 4,
                    5 * pi / 3,
                    3 * pi / 2,
                ]
                for i in thetas:
                    plt.plot(np.cos(i), np.sin(i), "k.")

        if n == 3:

            fig = plt.figure()

            points = 10000

            ax = fig.add_subplot(projection="3d")

            u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            if ticks is True:
                ax.plot_wireframe(x, y, abs(z), color="magenta", alpha=0.1)

            bunch_of_projections = self.random_uniform(points)

            for i in range(points):
                vector = np.random.rand(3, 1)
                x, y, z = np.matmul(bunch_of_projections[i], vector) / np.linalg.norm(
                    np.matmul(bunch_of_projections[i], vector)
                )
                # print("The coords are: ",x,y,z)
                ax.plot(x, y, abs(z), "sk", alpha=0.3)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 1])
            ax.view_init(10, 80)
            # Hide grid lines
            ax.grid(False)
            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.show()

    def plot_rendering(self, ticks):

        """
        Plots a Grassmannian manifold using the projection matricies
        generated by the Grassmannian class but with randomly sampled data.

        Works for cases with k=1 and n=2,3.
        Code will be generalized for future use.

        INPUT: Grassmannian manifold and True/False for ticks to determine
        if you want labels

        OUTPUT: None, plots manifold using MatPlotLib

        """

        n = self.n  # Size of space
        k = self.k  # Subspace size
        if k > 1 or n > 3:
            print(
                "Cannot handle sizes of k larger than 1 at this time.\
                 Cannot handle n larger than 3 at this time."
            )
            return -1

        # Plotting for 2d
        if n == 2:
            projections = self.random_uniform(100)
            # draw_points(self,projections,True)
            self.plot(ticks)
            plot_me = projection_to_two_d(projections, np.shape(projections)[0])
            for i in range(np.shape(plot_me)[0]):
                plt.plot(plot_me[i][0], plot_me[i][1], "y.")

        if n == 3:
            points = 10000

            point_draw = 100

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            if ticks is True:
                ax.plot_wireframe(x, y, abs(z), color="magenta", alpha=0.1)

            bunch_of_projections = self.random_uniform(points)
            bunch_two = self.random_uniform(point_draw)

            # plot manifold
            for i in range(points):
                vector = np.random.rand(3, 1)
                x, y, z = np.matmul(bunch_of_projections[i], vector) / np.linalg.norm(
                    np.matmul(bunch_of_projections[i], vector)
                )
                # print("The coords are: ",x,y,z)
                ax.plot(x, y, abs(z), "sk")

            # plot points on manifold
            for i in range(point_draw):
                vector = np.random.rand(3, 1)
                x, y, z = np.matmul(bunch_two[i], vector) / np.linalg.norm(
                    np.matmul(bunch_two[i], vector)
                )
                # print("The coords are: ",x,y,z)
                ax.plot(x, y, abs(z), "sm", alpha=0.3)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 1])
            ax.view_init(10, 80)
            # Hide grid lines
            ax.grid(False)
            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.show()

    def plot_tangent_space(self):

        """WIP DONT USE!!!!!! ONLY WORKS IN 2D BUT MIGHT NOT EVEN WORK RIGHT
        SO FOR FUTURE DIRECTIONS"""

        self.plot(True)
        projection = self.random_uniform(1)
        point = projection_to_two_d(projection, 1)

        print("Random point is: ", point[0], " ", point[1])

        plt.plot(point[0], point[1], "g+")

        normal_circle = [0, 0, 1]

        point_in_3d = [point[0], point[1], 0]

        tangent = np.cross(point_in_3d, normal_circle)

        x_part = [point[0], point[0] + tangent[0]]

        y_part = [point[1], point[1] + tangent[1]]

        plt.plot(x_part, y_part, "-g")


class GrassmannianCanonicalMetric(MatricesMetric, RiemannianMetric):
    """Canonical metric of the Grassmann manifold.

    Coincides with the Frobenius metric.

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    k : int
        Dimension of the subspaces.
    """

    def __init__(self, n, p):
        geomstats.errors.check_integer(p, "p")
        geomstats.errors.check_integer(n, "n")
        if p > n:
            raise ValueError("p <= n is required.")

        dim = int(p * (n - p))
        super(GrassmannianCanonicalMetric, self).__init__(
            m=n, n=n, dim=dim, signature=(dim, 0, 0)
        )

        self.n = n
        self.p = p
        self.embedding_metric = EuclideanMetric(n * p)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Exponentiate the invariant vector field v from base point p.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
            `tangent_vec` is the bracket of a skew-symmetric matrix with the
            base_point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        expm = gs.linalg.expm
        mul = Matrices.mul
        rot = Matrices.bracket(base_point, -tangent_vec)
        return mul(expm(rot), base_point, expm(-rot))

    def log(self, point, base_point, **kwargs):
        r"""Compute the Riemannian logarithm of point w.r.t. base_point.

        Given :math:`P, P'` in Gr(n, k) the logarithm from :math:`P`
        to :math:`P` is induced by the infinitesimal rotation [Batzies2015]_:

        .. math::

            Y = \frac 1 2 \log \big((2 P' - 1)(2 P - 1)\big)

        The tangent vector :math:`X` at :math:`P`
        is then recovered by :math:`X = [Y, P]`.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Riemannian logarithm, a tangent vector at `base_point`.

        References
        ----------
        .. [Batzies2015] Batzies, Hüper, Machado, Leite.
            "Geometric Mean and Geodesic Regression on Grassmannians"
            Linear Algebra and its Applications, 466, 83-101, 2015.
        """
        GLn = GeneralLinear(self.n)
        id_n = GLn.identity
        points = [id_n, point, base_point]
        id_n, point, base_point = gs.convert_to_wider_dtype(points)
        sym2 = 2 * point - id_n
        sym1 = 2 * base_point - id_n
        rot = GLn.compose(sym2, sym1)
        return Matrices.bracket(GLn.log(rot) / 2, base_point)

    def parallel_transport(
        self, tangent_vec, base_point, tangent_vec_b=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t\mapsto exp_(base_point)(
        t*direction)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n, n]
            Point on the Grassmann manifold. Point to transport from.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None
        end_point : array-like, shape=[..., n, n]
            Point on the Grassmann manifold to transport to.
            Unused if `tangent_vec_b` is given.
            Optional, default: None

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.

        References
        ----------
        .. [BZA20]  Bendokat, Thomas, Ralf Zimmermann, and P.-A. Absil.
                    “A Grassmann Manifold Handbook: Basic Geometry and
                    Computational Aspects.”
                    ArXiv:2011.13699 [Cs, Math], November 27, 2020.
                    https://arxiv.org/abs/2011.13699.

        """
        if tangent_vec_b is None:
            if end_point is not None:
                tangent_vec_b = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be \
                    given to define the geodesic along which to transport."
                )
        expm = gs.linalg.expm
        mul = Matrices.mul
        rot = -Matrices.bracket(base_point, tangent_vec_b)
        return mul(expm(rot), tangent_vec, expm(-rot))

    def private_squared_dist(self, point_a, point_b):
        """Compute geodesic distance between two points.

        Compute the squared geodesic distance between point_a
        and point_b, as defined by the metric.

        This is an auxiliary private function that:
        - is called by the method `squared_dist` of the class
        GrassmannianCanonicalMetric,
        - has been created to support the implementation
        of custom_gradient in tensorflow backend.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        _ : array-like, shape=[...,]
            Geodesic distance between point_a and point_b.
        """
        dist = super().squared_dist(point_a, point_b)
        return dist

    def squared_dist(self, point_a, point_b, **kwargs):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        dist = _squared_dist(point_a, point_b, metric=self)
        return dist
