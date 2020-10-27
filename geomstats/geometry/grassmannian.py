r"""
Manifold of linear subspaces.

The Grassmannian :math:`Gr(n, k)` is the manifold of k-dimensional
subspaces in n-dimensional Euclidean space.

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

"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-5
EPSILON = 1e-6


class Grassmannian(EmbeddedManifold):
    """Class for Grassmann manifolds Gr(n, k).

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    k : int
        Dimension of the subspaces.
    """

    def __init__(self, n, k):
        geomstats.errors.check_integer(k, 'k')
        geomstats.errors.check_integer(n, 'n')
        if k > n:
            raise ValueError(
                'k <= n is required: k-dimensional subspaces in n dimensions.')

        self.n = n
        self.k = k
        self.metric = GrassmannianCanonicalMetric(n, k)

        dim = int(k * (n - k))
        super(Grassmannian, self).__init__(
            dim=dim,
            embedding_manifold=Matrices(n, n),
            default_point_type='matrix')

        self.n = n
        self.k = k
        self.metric = GrassmannianCanonicalMetric(3, 2)

    def belongs(self, point, atol=TOLERANCE):
        """Check if the point belongs to the manifold.

        Check if an (n,n)-matrix is an orthogonal projector
        onto a subspace of rank k.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to be checked.
        atol : int
            Optional, default: 1e-5.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Grassmannian.
        """
        if not gs.all(self._check_square(point)):
            raise ValueError('all points must be square.')

        symm = Matrices.is_symmetric(point)
        idem = self._check_idempotent(point, atol)
        rank = self._check_rank(point, self.k, atol)

        belongs = gs.all(gs.stack([symm, idem, rank], axis=0), axis=0)

        return belongs

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
            points,
            GeneralLinear.inverse(full_rank),
            Matrices.transpose(points))
        return projector[0] if n_samples == 1 else projector

    def is_tangent(self, vector, base_point=None, atol=TOLERANCE):
        r"""Check if a vector is tangent to the manifold at the base point.

        Check if the (n,n)-matrix :math: `Y` is symmetric and verifies the
        relation :math: PY + YP = Y where :math: `P` represents the base
        point and :math: `Y` the vector.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to be checked.
        base_point : array-like, shape=[..., n, n]
            Base point.
        atol : int
            Optional, default: 1e-5.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if `vector` is tangent to the Grassmannian at
            `base_point`.
        """
        diff = Matrices.mul(
            base_point, vector) + Matrices.mul(vector, base_point) - vector
        is_close = gs.all(gs.isclose(diff, 0., atol=atol))
        return gs.logical_and(Matrices.is_symmetric(vector), is_close)

    def to_tangent(self, vector, base_point=None):
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
        skew = Matrices.to_skew_symmetric(vector)
        return Matrices.bracket(base_point, skew)

    @staticmethod
    def _check_square(point):
        """Check if a point is a square matrix.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        belongs : bool
        """
        n, p = point.shape[-2:]

        return n == p

    @staticmethod
    def _check_idempotent(point, tolerance):
        """Check that a point is idempotent.

        Parameters
        ----------
        point
        tolerance

        Returns
        -------
        belongs : bool
        """
        diff = gs.einsum('...ij,...jk->...ik', point, point) - point
        diff_norm = gs.linalg.norm(diff, axis=(-2, -1))

        return gs.less_equal(diff_norm, tolerance)

    @staticmethod
    def _check_rank(point, rank, tolerance):
        """Check rank of a matrix.

        Check that the rank of the point is equal to the
        subspace dimension.  Matrix rank is equal to number of
        singular values greater than 0.

        Parameters
        ----------
        point
        rank
        tolerance

        Returns
        -------
        belongs : bool
        """
        [_, s, _] = gs.linalg.svd(point)

        return gs.sum(s > tolerance, axis=-1) == rank


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
        geomstats.errors.check_integer(p, 'p')
        geomstats.errors.check_integer(n, 'n')
        if p > n:
            raise ValueError('p <= n is required.')

        dim = int(p * (n - p))
        super(GrassmannianCanonicalMetric, self).__init__(
            m=n, n=n, dim=dim, signature=(dim, 0, 0))

        self.n = n
        self.p = p
        self.embedding_metric = EuclideanMetric(n * p)

    def exp(self, tangent_vec, base_point):
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

    def log(self, point, base_point):
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
        id_n, point, base_point = gs.convert_to_wider_dtype([
            id_n, point, base_point])
        sym2 = 2 * point - id_n
        sym1 = 2 * base_point - id_n
        rot = GLn.mul(sym2, sym1)
        return Matrices.bracket(GLn.log(rot) / 2, base_point)
