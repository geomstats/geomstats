r"""
Manifold of linear subspaces.

The Grassmannian :math:`Gr(n, k)` is the manifold of k-dimensional
subspaces in n-dimensional Euclidean space.

Lead author: Olivier Peltre.

:math:`Gr(n, p)` is represented by
:math:`n \times n` matrices
of rank :math:`p`  satisfying :math:`P^2 = P` and :math:`P^T = P`.
Each :math:`P \in Gr(n, p)` is identified with the unique
orthogonal projector onto :math:`{\rm Im}(P)`.

:math:`Gr(n, p)` is a homogoneous space, quotient of the special orthogonal
group by the subgroup of rotations stabilising a :math:`p`-dimensional subspace:

.. math::

    Gr(n, p) \simeq \frac {SO(n)} {SO(p) \times SO(n-p)}

It is therefore customary to represent the Grassmannian
by equivalence classes of orthogonal :math:`p`-frames in :math:`{\mathbb R}^n`.
For such a representation, work in the Stiefel manifold instead.

.. math::

    Gr(n, p) \simeq St(n, p) / SO(p)

References
----------
.. [Batzies15] Batzies, E., K. Hüper, L. Machado, and F. Silva Leite.
    “Geometric Mean and Geodesic Regression on Grassmannians.”
    Linear Algebra and Its Applications 466 (February 1, 2015):
    83–101. https://doi.org/10.1016/j.laa.2014.10.003.
"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import LevelSet
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.vectorization import repeat_out


class Grassmannian(LevelSet):
    """Class for Grassmann manifolds :math:`Gr(n, p)`.

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    p : int
        Dimension of the subspaces.
    """

    def __init__(self, n, p, equip=True):
        geomstats.errors.check_integer(p, "p")
        geomstats.errors.check_integer(n, "n")
        if p > n:
            raise ValueError(
                "p < n is required: p-dimensional subspaces in n dimensions."
            )

        self.n = n
        self.p = p

        dim = int(p * (n - p))
        super().__init__(dim=dim, equip=equip)

    def _define_embedding_space(self):
        return SymmetricMatrices(self.n)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return GrassmannianCanonicalMetric

    def submersion(self, point):
        r"""Submersion that defines the Grassmann manifold.

        The Grassmann manifold is defined here as embedded in the set of
        symmetric matrices, as the pre-image of the function defined around the
        projector on the space spanned by the first :math:`p` columns of the identity
        matrix by (see Exercise E.25 in [Pau07]_).

        .. math::

            \begin{pmatrix} I_p + A & B^T \\ B & D \end{pmatrix} \mapsto
                (D - B(I_p + A)^{-1}B^T, A + A^2 + B^TB)

        This map is a submersion and its zero space is the set of orthogonal
        rank-:math:`p` projectors.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n, n]

        References
        ----------
        .. [Pau07] Paulin, Frédéric. “Géométrie diﬀérentielle élémentaire,” 2007.
            https://www.imo.universite-paris-saclay.fr/~paulin/notescours/
            cours_geodiff.pdf.
        """
        p = self.p

        _, eigvecs = gs.linalg.eigh(point)
        eigvecs = gs.flip(eigvecs, -1)
        flipped_point = Matrices.mul(Matrices.transpose(eigvecs), point, eigvecs)
        b = flipped_point[..., p:, :p]
        d = flipped_point[..., p:, p:]
        a = flipped_point[..., :p, :p] - gs.eye(p)
        first = d - Matrices.mul(
            b, GeneralLinear.inverse(a + gs.eye(p)), Matrices.transpose(b)
        )
        second = a + Matrices.mul(a, a) + Matrices.mul(Matrices.transpose(b), b)
        row_1 = gs.concatenate([first, gs.zeros_like(b)], axis=-1)
        row_2 = gs.concatenate([Matrices.transpose(gs.zeros_like(b)), second], axis=-1)
        return gs.concatenate([row_1, row_2], axis=-2)

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_vector : array-like, shape=[..., n, n]
        """
        return 2 * Matrices.to_symmetric(Matrices.mul(point, vector)) - vector

    def random_uniform(self, n_samples=1):
        r"""Sample random points from a uniform distribution.

        Following [Chikuse03]_, :math:`n\_samples * n * p` scalars are sampled
        from a standard normal distribution and reshaped to matrices,
        the projectors on their first :math:`p` columns follow a uniform
        distribution.

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
        points = gs.random.normal(size=(n_samples, self.n, self.p))
        full_rank = Matrices.mul(Matrices.transpose(points), points)
        projector = Matrices.mul(
            points, GeneralLinear.inverse(full_rank), Matrices.transpose(points)
        )
        return projector[0] if n_samples == 1 else projector

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample random points from a uniform distribution.

        Following [Chikuse03]_, :math:`n\_samples * n * p` scalars are sampled
        from a standard normal distribution and reshaped to matrices,
        the projectors on their first :math:`p` columns follow a uniform
        distribution.

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
        diagonal = gs.array([0.0] * (self.n - self.p) + [1.0] * self.p)
        p_d = gs.einsum("...ij,...j->...ij", eigvecs, diagonal)
        return Matrices.mul(p_d, Matrices.transpose(eigvecs))


class GrassmannianCanonicalMetric(RiemannianMetric):
    """Canonical metric of the Grassmann manifold."""

    def __init__(self, space):
        super().__init__(space=space, signature=(space.dim, 0, 0))
        self._general_linear = GeneralLinear(space.n, equip=False)

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

        Given :math:`P, P'` in :math:`Gr(n, p)` the logarithm from :math:`P`
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
        id_n = self._general_linear.identity
        id_n, point, base_point = gs.convert_to_wider_dtype([id_n, point, base_point])
        sym2 = 2 * point - id_n
        sym1 = 2 * base_point - id_n
        rot = self._general_linear.compose(sym2, sym1)
        return Matrices.bracket(self._general_linear.log(rot) / 2, base_point)

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n, n]
            Point on the Grassmann manifold. Point to transport from.
        direction : array-like, shape=[..., n, n]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None
        end_point : array-like, shape=[..., n, n]
            Point on the Grassmann manifold to transport to. Unused if
            `direction` is given.
            Optional, default: None

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at `exp_(base_point)(direction)`.

        References
        ----------
        .. [BZA20]  Bendokat, Thomas, Ralf Zimmermann, and P.-A. Absil.
            “A Grassmann Manifold Handbook: Basic Geometry and Computational
            Aspects.” ArXiv:2011.13699 [Cs, Math], November 27, 2020.
            https://arxiv.org/abs/2011.13699.
        """
        if direction is None:
            if end_point is not None:
                direction = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a direction must be given to define the"
                    " geodesic along which to transport."
                )
        expm = gs.linalg.expm
        mul = Matrices.mul
        rot = -Matrices.bracket(base_point, direction)
        return mul(expm(rot), tangent_vec, expm(-rot))

    def squared_dist(self, point_a, point_b):
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
        sdist_func = super().squared_dist

        def _squared_dist_grad_point_a(point_a, point_b):
            """Compute gradient of squared_dist wrt point_a."""
            return -2 * self.log(point_b, point_a)

        def _squared_dist_grad_point_b(point_a, point_b):
            """Compute gradient of squared_dist wrt point_b."""
            return -2 * self.log(point_a, point_b)

        @gs.autodiff.custom_gradient(
            _squared_dist_grad_point_a, _squared_dist_grad_point_b
        )
        def _squared_dist(point_a, point_b):
            """Compute geodesic distance between two points.

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
            return sdist_func(point_a, point_b)

        return _squared_dist(point_a, point_b)

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In this case it is Pi / 2 everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.

        References
        ----------
        .. [BZA20]  Bendokat, Thomas, Ralf Zimmermann, and P.-A. Absil.
            “A Grassmann Manifold Handbook: Basic Geometry and
            Computational Aspects.”
            ArXiv:2011.13699 [Cs, Math], November 27, 2020.
            https://arxiv.org/abs/2011.13699.
        """
        radius = gs.array(gs.pi / 2)
        return repeat_out(self._space.point_ndim, radius, base_point)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute Frobenius inner-product of two tangent vectors.

        Coincides with the Frobenius metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., m, n]
            Tangent vector.
        tangent_vec_b : array-like, shape=[..., m, n]
            Tangent vector.
        base_point : array-like, shape=[..., m, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Frobenius inner-product of tangent_vec_a and tangent_vec_b.
        """
        inner_prod = Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)
        return repeat_out(
            self._space.point_ndim, inner_prod, tangent_vec_a, tangent_vec_b, base_point
        )

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        sq_norm = gs.linalg.norm(vector, axis=(-2, -1)) ** 2
        return repeat_out(self._space.point_ndim, sq_norm, vector, base_point)

    def norm(self, vector, base_point=None):
        """Compute norm of a matrix.

        Norm of a matrix associated to the Frobenius inner product.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        norm = gs.linalg.norm(vector, axis=(-2, -1))
        return repeat_out(self._space.point_ndim, norm, vector, base_point)
