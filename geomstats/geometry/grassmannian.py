"""
Manifold of linear subspaces.

The Grassmannian :math:`Gr(n, k)` is the manifold of k-dimensional
subspaces in n-dimensional Euclidean space.

:math:`Gr(n, k)` is represented by
:math:`n \\times n` matrices
of rank :math:`k`  satisfying :math:`P^2 = P` and :math:`P^T = P`.
Each :math:`P \in Gr(n, k)` is identified with the unique
orthogonal projector onto :math:`{\\rm Im}(P)`.

:math:`Gr(n, k)` is a homogoneous space, quotient of the special orthogonal group
by the subgroup of rotations stabilising a k-dimensional subspace:

.. math::

    Gr(n, k) \simeq \\frac {SO(n)} {SO(k) \\times SO(n-k)}

It is therefore customary to represent the Grassmannian
by equivalence classes of orthogonal :math:`k`-frames in :math:`{\\mathbb R}^n`.
For such a representation, work in the Stiefel manifold instead.

.. math::

    Gr(n, k) \simeq St(n, k) / SO(k)

"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
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
        self.metric = GrassmannianCanonicalMetric(3, 2)

        dim = int(k * (n - k))
        super(Grassmannian, self).__init__(
            dim=dim,
            embedding_manifold=Matrices(n, n),
            default_point_type='matrix')

        self.n = n
        self.k = k
        self.metric = GrassmannianCanonicalMetric(3, 2)

    def belongs(self, point, tolerance=TOLERANCE):
        """Check if an (n,n)-matrix is a rank-k projector.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to be checked.
        tolerance : int
            Optional, default: 1e-5.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Grassmannian.
        """
        raise NotImplementedError(
            'The Grassmann `belongs` is not implemented.'
            'It shall test whether p*=p, p^2 = p and rank(p) = k.')

    def is_tangent(self, tangent_vec, point):
        """Check if an (n,n)-matrix is tangent to a point in the Grassmannian.

        A matrix :math:`X` is tangent to :math:`P \\in Gr(n, k)`
        if and only if :math:`X^T = X` and :math:`PX + XP = X`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector to be checked.
        point : array-like, shape=[..., n, n]
            Base point.

        Returns
        _______
        is_tangent : boolean, shape=[...]

        """
        is_inf_rot = Matrices.is_skew_symmetric(tangent_vec)
        is_transverse = Matrices.equal(
                Matrices.bracket(tangent_vec, point),
                point)
        return gs.logical_and(is_inf_rot, is_transverse)

    def to_tangent(self, tangent_vec, point):
        inf_rot = Matrices.to_skew_symmetric(tangent_vec)
        return Matrices.bracket(inf_rot, point)

class GrassmannianCanonicalMetric(RiemannianMetric):
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
            dim=dim,
            signature=(dim, 0, 0))

        self.n = n
        self.p = p
        self.embedding_metric = EuclideanMetric(n * p)

    def exp(self, tangent_vec, base_point):
        """Exponentiate the invariant vector field v from base point p.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        expm = gs.linalg.expm
        mul = Matrices.mul
        return mul(expm(tangent_vec), base_point, expm(-tangent_vec))

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
        sym2 = 2 * point - id_n
        sym1 = 2 * base_point - id_n
        rot = GLn.mul(sym2, sym1)
        return GLn.log(rot) / 2
