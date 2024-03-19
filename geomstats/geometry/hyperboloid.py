"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded with
the hyperboloid representation (embedded in minkowsky space).

Lead author: Nina Miolane.
"""

import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry._hyperbolic import HyperbolicMetric, _Hyperbolic
from geomstats.geometry.base import LevelSet
from geomstats.geometry.minkowski import Minkowski
from geomstats.vectorization import repeat_out


class Hyperboloid(_Hyperbolic, LevelSet):
    """Class for the n-dimensional hyperboloid space.

    Class for the n-dimensional hyperboloid space as embedded in (n+1)-dimensional
    Minkowski space as the set of points with squared norm equal to -1, i.e.
    .. math::
        - x_0^2 + x_1^2 + ... + x_n^2 = - 1.

    For other
    representations of hyperbolic spaces see the `Hyperbolic` class.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    """

    def __init__(self, dim, equip=True):
        self.coords_type = "extrinsic"
        self.dim = dim
        super().__init__(dim=dim, intrinsic=False, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return HyperboloidMetric

    def _define_embedding_space(self):
        return Minkowski(self.dim + 1)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]

        Returns
        -------
        submersed_point : array-like, shape=[...]
        """
        return self.embedding_space.metric.squared_norm(point) + 1.0

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
        point : array-like, shape=[..., dim + 1]

        Returns
        -------
        submersed_vector : array-like, shape=[...]
        """
        return self.embedding_space.metric.inner_product(vector, point)

    def projection(self, point):
        """Project a point in space on the hyperboloid.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point projected on the hyperboloid.
        """
        belongs = self.belongs(point)

        # avoid dividing by 0
        factor = gs.where(point[..., 0] == 0.0, 1.0, point[..., 0] + gs.atol)

        first_coord = gs.where(belongs, 1.0, 1.0 / factor)
        intrinsic = gs.einsum("...,...i->...i", first_coord, point)[..., 1:]
        return self.intrinsic_to_extrinsic_coords(intrinsic)

    def regularize(self, point):
        """Regularize a point to the canonical representation.

        Regularize a point to the canonical representation chosen
        for the hyperbolic space, to avoid numerical issues.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in canonical representation
            in extrinsic coordinates.
        """
        sq_norm = self.embedding_space.metric.squared_norm(point)
        if not gs.all(sq_norm):
            raise ValueError(
                "Cannot project a vector of norm 0. in the "
                "Minkowski space to the hyperboloid"
            )
        real_norm = gs.sqrt(gs.abs(sq_norm))
        return gs.einsum("...i,...->...i", point, 1.0 / real_norm)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the hyperbolic space.

        Project a vector in Minkowski space on the tangent space
        of the hyperbolic space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector in Minkowski space to be projected.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at the base point, equal to the projection of
            the vector in Minkowski space.
        """
        sq_norm = self.embedding_space.metric.squared_norm(base_point)
        inner_prod = self.embedding_space.metric.inner_product(base_point, vector)

        coef = inner_prod / sq_norm

        return vector - gs.einsum("...,...j->...j", coef, base_point)

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in the embedded manifold in extrinsic coordinates.
        """
        return self.change_coordinates_system(point_intrinsic, "intrinsic", "extrinsic")

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.

        Returns
        -------
        point_intrinsic : array-like, shape=[..., dim]
            Point in intrinsic coordinates.
        """
        return self.change_coordinates_system(point_extrinsic, "extrinsic", "intrinsic")

    def project_on_geodesic(self, point, mean, vector):
        """Project on geodesic in extrinsic coordinates.

        Project point onto geodesic going through mean in direction of vector.
        See reference below.

        Parameters
        ----------
        point: array-like, shape=[..., dim + 1]
            Point in hyperbolic space.
        mean: array-like, shape=[..., dim + 1]
            Point through which the geodesic passes.
        vector : array-like, shape=[..., dim + 1]
            Vector in Minkowski space, direction of the geodesic.

        Returns
        -------
        proj : array-like, shape=[..., dim + 1]
            Projected point on the geodesic.

        References
        ----------
        .. [CSV2016] R. Chakraborty, D. Seo, and B. C. Vemuri,
            "An efficient exact-pga algorithm for constant curvature manifolds."
            Proceedings of the IEEE conference on computer vision and pattern
            recognition. 2016.
        """
        inner_prod_1 = self.metric.inner_product(point, vector)
        inner_prod_2 = self.metric.inner_product(point, mean)
        norm_v = self.metric.norm(vector, mean)
        dist_to_proj = gs.arctanh(-inner_prod_1 / inner_prod_2 / norm_v)
        proj = gs.einsum("...,i->...i", gs.cosh(dist_to_proj), mean) + gs.einsum(
            "...,i->...i", gs.sinh(dist_to_proj), vector / norm_v
        )
        return proj


class HyperboloidMetric(HyperbolicMetric):
    """Class that defines operations using a hyperbolic metric."""

    def metric_matrix(self, base_point=None):
        """Compute the inner product matrix.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim + 1]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod_mat: array-like, shape=[..., dim+1, dim + 1]
            Inner-product matrix.
        """
        return self._space.embedding_space.metric.metric_matrix(base_point)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point in hyperbolic space.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        return self._space.embedding_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector on the tangent space of the hyperbolic space at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm of the vector.
        """
        return self._space.embedding_space.metric.squared_norm(vector)

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        exp : array-like, shape=[..., dim + 1]
            Point in hyperbolic space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        sq_norm_tangent_vec = self._space.embedding_space.metric.squared_norm(
            tangent_vec
        )
        sq_norm_tangent_vec = gs.clip(sq_norm_tangent_vec, 0, math.inf)

        coef_1 = utils.taylor_exp_even_func(
            sq_norm_tangent_vec, utils.cosh_close_0, order=5
        )
        coef_2 = utils.taylor_exp_even_func(
            sq_norm_tangent_vec, utils.sinch_close_0, order=5
        )

        exp = gs.einsum("...,...j->...j", coef_1, base_point) + gs.einsum(
            "...,...j->...j", coef_2, tangent_vec
        )

        return self._space.regularize(exp)

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        If `coords_type` is `poincare` then base_point belongs
        to the Poincare ball and point is a vector in the Euclidean
        space of the same dimension as the ball.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        angle = self.dist(base_point, point)

        coef_1_ = utils.taylor_exp_even_func(angle**2, utils.inv_sinch_close_0, order=4)
        coef_2_ = utils.taylor_exp_even_func(angle**2, utils.inv_tanh_close_0, order=4)

        log_term_1 = gs.einsum("...,...j->...j", coef_1_, point)
        log_term_2 = -gs.einsum("...,...j->...j", coef_2_, base_point)
        return log_term_1 + log_term_2

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
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim + 1]
            First point in hyperbolic space.
        point_b : array-like, shape=[..., dim + 1]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[...,]
            Geodesic distance between the two points.
        """
        embedding_metric = self._space.embedding_space.metric
        sq_norm_a = embedding_metric.squared_norm(point_a)
        sq_norm_b = embedding_metric.squared_norm(point_b)
        inner_prod = embedding_metric.inner_product(point_a, point_b)

        cosh_angle = -inner_prod / gs.sqrt(sq_norm_a * sq_norm_b)
        cosh_angle = gs.clip(cosh_angle, 1.0, 1e24)

        return gs.arccosh(cosh_angle)

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
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hyperboloid.
        direction : array-like, shape=[..., dim + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default : None.
        end_point : array-like, shape=[..., dim + 1]
            Point on the hyperboloid. Point to transport to. Unused if `tangent_vec_b`
            is given.
            Optional, default : None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim + 1]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        if direction is None:
            if end_point is not None:
                direction = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be given to define the"
                    " geodesic along which to transport."
                )
        theta = self._space.embedding_space.metric.norm(direction)
        eps = gs.where(theta == 0.0, 1.0, theta)
        normalized_b = gs.einsum("...,...i->...i", 1 / eps, direction)
        pb = self._space.embedding_space.metric.inner_product(tangent_vec, normalized_b)
        p_orth = tangent_vec - gs.einsum("...,...i->...i", pb, normalized_b)
        transported = (
            gs.einsum("...,...i->...i", gs.sinh(theta) * pb, base_point)
            + gs.einsum("...,...i->...i", gs.cosh(theta) * pb, normalized_b)
            + p_orth
        )
        return transported

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In the case of the hyperbolic space, it does not depend on the base
        point and is infinite everywhere, because of the negative curvature.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim+1]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)
