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

References
----------
[Batzies15]_    Batzies, E., K. Hüper, L. Machado, and F. Silva Leite.
                “Geometric Mean and Geodesic Regression on Grassmannians.”
                Linear Algebra and Its Applications 466 (February 1, 2015):
                83–101. https://doi.org/10.1016/j.laa.2014.10.003.
"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


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
        b, GeneralLinear.inverse(a + gs.eye(k)), Matrices.transpose(b))
    second = a + Matrices.mul(a, a) + Matrices.mul(Matrices.transpose(b), b)
    row_1 = gs.concatenate([first, gs.zeros_like(b)], axis=-1)
    row_2 = gs.concatenate([
        Matrices.transpose(gs.zeros_like(b)), second], axis=-1)
    return gs.concatenate([row_1, row_2], axis=-2)


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

        dim = int(k * (n - k))
        super(Grassmannian, self).__init__(
            dim=dim, embedding_space=SymmetricMatrices(n),
            submersion=lambda x: submersion(x, k), value=gs.zeros((n, n)),
            tangent_submersion=lambda v, x: 2 * Matrices.to_symmetric(
                Matrices.mul(x, v)) - v,
            metric=GrassmannianCanonicalMetric(n, k))

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

    def random_point(self, n_samples=1, bound=1.):
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
        diagonal = gs.array([0.] * (self.n - self.k) + [1.] * self.k)
        p_d = gs.einsum('...ij,...j->...ij', eigvecs, diagonal)
        return Matrices.mul(p_d, Matrices.transpose(eigvecs))


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
        id_n, point, base_point = gs.convert_to_wider_dtype([
            id_n, point, base_point])
        sym2 = 2 * point - id_n
        sym1 = 2 * base_point - id_n
        rot = GLn.compose(sym2, sym1)
        return Matrices.bracket(GLn.log(rot) / 2, base_point)

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by :math: `t \mapsto exp_(base_point)(t*
        tangent_vec_b)`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point, along which the parallel transport
            is computed.
        base_point : array-like, shape=[..., n, n]
            Point on the Grassmann manifold.

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
                    http://arxiv.org/abs/2011.13699.

        """
        expm = gs.linalg.expm
        mul = Matrices.mul
        rot = Matrices.bracket(base_point, -tangent_vec_b)
        return mul(expm(rot), tangent_vec_a, expm(-rot))
