"""The n-dimensional hyperbolic space.

Poincare half-space representation.

Lead author: Alice Le Brigant.
"""

import math

import geomstats.backend as gs
from geomstats.geometry._hyperbolic import HyperbolicDiffeo, _Hyperbolic
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.pullback_metric import CoordinatePullbackDiffeoMetric
from geomstats.vectorization import get_batch_shape, repeat_out


class PoincareHalfSpace(_Hyperbolic, VectorSpaceOpenSet):
    """Class for the n-dimensional Poincare half-space.

    Class for the n-dimensional Poincaré half space model. For other
    representations of hyperbolic spaces see the `Hyperbolic` class.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    """

    def __init__(self, dim, equip=True):
        self.coords_type = "half-space"
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim),
            intrinsic=True,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PoincareHalfSpaceMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the upper half space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be checked.
        atol : float
            Absolute tolerance to evaluate positivity of the last coordinate.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Array of booleans indicating whether the corresponding
            points belong to the upper half space.
        """
        return point[..., -1] > -atol

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The last coordinate is floored to `atol` if it is negative.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[..., dim_embedding]
            Projected point.
        """
        last = gs.where(point[..., -1] < atol, atol, point[..., -1])
        return gs.concatenate([point[..., :-1], last[..., None]], axis=-1)


class PoincareHalfSpaceMetric(CoordinatePullbackDiffeoMetric):
    r"""Class for the metric of the n-dimensional hyperbolic space.

    Class for the metric of the n-dimensional hyperbolic space
    as embedded in the Poincaré half space model.

    It is the conformal metric :math:`g = \delta / y^2`, with :math:`y` the
    last coordinate, and has constant sectional curvature :math:`-1`.
    """

    def __init__(self, space):
        image_space = PoincareBall(dim=space.dim)
        super().__init__(
            space=space,
            image_space=image_space,
            diffeo=HyperbolicDiffeo(space.coords_type, image_space.coords_type),
        )

    def _christoffels_pattern(self, dtype):
        r"""Compute the constant part of the Christoffel symbols.

        It is :math:`\delta_{ik} e_j + \delta_{jk} e_i - \delta_{ij} e_k`,
        with :math:`e` the last vector of the canonical basis.

        Parameters
        ----------
        dtype : dtype
            Data type.

        Returns
        -------
        pattern : array-like, shape=[dim, dim, dim]
            Pattern, with the contravariant index on the first dimension.
        """
        eye = gs.eye(self._space.dim, dtype=dtype)
        last = eye[-1]
        return (
            gs.einsum("ki,j->kij", eye, last)
            + gs.einsum("kj,i->kij", eye, last)
            - gs.einsum("ij,k->kij", eye, last)
        )

    def metric_matrix(self, base_point):
        """Compute the inner product matrix at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner product matrix.
        """
        eye = gs.eye(self._space.dim, dtype=base_point.dtype)
        return gs.einsum("...,ij->...ij", 1 / base_point[..., -1] ** 2, eye)

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        It is :math:`\partial_k g_{ij} = -2 \delta_{ij} \delta_{kn} / y^3`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        metric_derivative : array-like, shape=[..., dim, dim, dim]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        eye = gs.eye(self._space.dim, dtype=base_point.dtype)
        return gs.einsum(
            "...,ij,k->...ijk", -2 / base_point[..., -1] ** 3, eye, eye[-1]
        )

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        They are :math:`\Gamma^k_{ij} = -(\delta_{ik} \delta_{jn}
        + \delta_{jk} \delta_{in} - \delta_{ij} \delta_{kn}) / y`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """
        pattern = self._christoffels_pattern(base_point.dtype)
        return gs.einsum("...,kij->...kij", -1 / base_point[..., -1], pattern)

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        It is :math:`\partial_l \Gamma^k_{ij} = (\delta_{ik} \delta_{jn}
        + \delta_{jk} \delta_{in} - \delta_{ij} \delta_{kn}) \delta_{ln} / y^2`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., dim, dim, dim, dim]
            Jacobian of the Christoffel symbols.
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`
        """
        pattern = self._christoffels_pattern(base_point.dtype)
        last = gs.eye(self._space.dim, dtype=base_point.dtype)[-1]
        return gs.einsum(
            "...,kij,l->...kijl", 1 / base_point[..., -1] ** 2, pattern, last
        )

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point=None):
        r"""Compute the Riemann curvature map R.

        The hyperbolic space has constant curvature :math:`-1`, so
        :math:`R(X, Y)Z = - (\langle Y, Z \rangle X - \langle X, Z \rangle Y)`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        tangent_vec_c : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        curvature : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        """
        inner_prod_bc = self.inner_product(tangent_vec_b, tangent_vec_c, base_point)
        inner_prod_ac = self.inner_product(tangent_vec_a, tangent_vec_c, base_point)
        return gs.einsum("...,...i->...i", inner_prod_ac, tangent_vec_b) - gs.einsum(
            "...,...i->...i", inner_prod_bc, tangent_vec_a
        )

    def curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point=None,
    ):
        """Compute the covariant derivative of the curvature.

        The hyperbolic space has constant curvature, so the covariant
        derivative of the curvature vanishes.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        tangent_vec_c : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        tangent_vec_d : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., dim]
            Tangent vector at ``base_point``.
        """
        batch_shape = get_batch_shape(
            self._space.point_ndim,
            tangent_vec_a,
            tangent_vec_b,
            tangent_vec_c,
            tangent_vec_d,
            base_point,
        )
        return gs.zeros(batch_shape + self._space.shape)

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the sectional curvature.

        The hyperbolic space has constant sectional curvature :math:`-1`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., dim]
            Point in the manifold.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.
        """
        return repeat_out(
            self._space.point_ndim,
            gs.array(-1.0),
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )

    def scalar_curvature(self, base_point=None):
        """Compute the scalar curvature.

        It is :math:`-n (n - 1)`, with :math:`n` the dimension.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point in the manifold.

        Returns
        -------
        scalar_curvature : array-like, shape=[...,]
            Scalar curvature at `base_point`.
        """
        dim = self._space.dim
        return repeat_out(
            self._space.point_ndim, gs.array(-dim * (dim - 1.0)), base_point
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        inner_prod : array-like, shape=[..., 1]
            Inner-product of the two tangent vectors.
        """
        inner_prod = gs.sum(tangent_vec_a * tangent_vec_b, axis=-1)
        return inner_prod / base_point[..., -1] ** 2

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
        return self.dist(point_a, point_b) ** 2

    def dist(self, point_a, point_b):
        r"""Compute the geodesic distance between two points.

        .. math::

            d(a, b) = \mathrm{arccosh}\left(1 + \frac{||a - b||^2}
            {2 a_n b_n}\right)

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Geodesic distance between the two points.
        """
        diff_norm = gs.sum((point_a - point_b) ** 2, axis=-1)
        return gs.arccosh(
            1 + diff_norm / (2 * point_a[..., -1] * point_b[..., -1])
        )

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In the case of the hyperbolic space, it does not depend on the base
        point and is infinite everywhere, because of the negative curvature.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)
