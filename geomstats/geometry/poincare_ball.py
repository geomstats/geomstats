"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space with Poincare ball model.

Lead author: Hadi Zaatiti.
"""

import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out

EPSILON = 1e-6
NORMALIZATION_FACTOR_CST = gs.sqrt(gs.pi / 2)
PI_2_3 = gs.power(gs.array([2.0 * gs.pi]), gs.array([2 / 3]))
SQRT_2 = gs.sqrt(2.0)

_COORDS_TYPE = "ball"


class PoincareBall(_Hyperbolic, OpenSet):
    """Class for the n-dimensional Poincare ball.

    Class for the n-dimensional Poincaré ball model. For other
    representations of hyperbolic spaces see the `Hyperbolic` class.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    """

    def __init__(self, dim, equip=True):
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim),
            default_coords_type=_COORDS_TYPE,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PoincareBallMetric

    def belongs(self, point, atol=gs.atol):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be tested.
        atol : float, optional
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        return gs.sum(point**2, axis=-1) < (1 - atol)

    def projection(self, point):
        """Project a point on the ball.

        Project a point by clipping such that l2
        norm being lower than 1

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point projected on the ball.
        """
        if point.shape[-1] != self.dim:
            raise NameError("Wrong dimension, expected ", self.dim)

        l2_norm = gs.linalg.norm(point, axis=-1)
        if gs.any(l2_norm >= 1 - gs.atol):
            projected_point = gs.einsum(
                "...j,...->...j", point * (1 - gs.atol), 1.0 / l2_norm
            )
            return -gs.maximum(-projected_point, -point)

        return gs.copy(point)


class PoincareBallMetric(RiemannianMetric):
    """Class that defines operations using a Poincare ball."""

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point in the Poincare ball.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in the Poincare ball equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        squared_norm_bp = gs.sum(base_point**2, axis=-1)
        norm_tan = gs.linalg.norm(tangent_vec, axis=-1)
        lambda_base_point = 1 / (1 - squared_norm_bp)

        # This avoids dividing by 0
        norm_tan_eps = gs.where(gs.isclose(norm_tan, 0.0), EPSILON, norm_tan)
        direction = gs.einsum("...i,...->...i", tangent_vec, 1 / norm_tan_eps)

        factor = gs.tanh(lambda_base_point * norm_tan)

        return self.mobius_add(
            base_point, gs.einsum("...,...i->...i", factor, direction)
        )

    def log(self, point, base_point, **kwargs):
        """Compute Riemannian logarithm of a point wrt a base point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the Poincare ball.
        base_point : array-like, shape=[..., dim]
            Point in the Poincare ball.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        mobius_addition = self.mobius_add(-base_point, point)
        squared_norm_add = gs.sum(mobius_addition**2, axis=-1)
        squared_norm_bp = gs.sum(base_point**2, axis=-1)
        coef = (1 - squared_norm_bp) * utils.taylor_exp_even_func(
            squared_norm_add, utils.arctanh_card_close_0
        )
        return gs.einsum("...,...j->...j", coef, mobius_addition)

    def mobius_add(self, point_a, point_b, project_first=True):
        r"""Compute the Mobius addition of two points.

        The Mobius addition is useful to compute the log and exp in the
        'ball' representation.

        .. math::

            a\oplus b=\frac{(1+2\langle a,b\rangle + ||b||^2)a+
            (1-||a||^2)b}{1+2\langle a,b\rangle + ||a||^2||b||^2}

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point in Poincare ball.
        point_b : array-like, shape=[..., dim]
            Point in Poincare ball.
        project_first : boolean
            Project points on the ball or not (according to tolerance).

        Returns
        -------
        mobius_add : array-like, shape=[..., dim]
            Result of the Mobius addition.
        """
        if project_first:
            point_a = self._space.projection(point_a)
            point_b = self._space.projection(point_b)
        else:
            point_a_belong = self._space.belongs(point_a)
            point_b_belong = self._space.belongs(point_b)

            if not gs.all(point_a_belong) or not gs.all(point_b_belong):
                raise ValueError("Points do not belong to the Poincare ball")

        inner = gs.sum(point_a * point_b, axis=-1)
        squared_norm_a = gs.sum(point_a**2, axis=-1)
        squared_norm_b = gs.sum(point_b**2, axis=-1)
        num_1 = gs.einsum("...,...i->...i", 1 + 2 * inner + squared_norm_b, point_a)
        num_2 = gs.einsum("...,...i->...i", 1 - squared_norm_a, point_b)
        result = gs.einsum(
            "...,...i->...i",
            1.0 / (1 + 2 * inner + squared_norm_b * squared_norm_a),
            num_1 + num_2,
        )
        return self._space.projection(result)

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            First point in the Poincare ball.
        point_b : array-like, shape=[..., dim]
            Second point in the Poincare ball.

        Returns
        -------
        dist : array-like, shape=[...,]
            Geodesic distance between the two points.
        """
        point_a_norm = gs.clip(gs.sum(point_a**2, -1), 0.0, 1 - EPSILON)
        point_b_norm = gs.clip(gs.sum(point_b**2, -1), 0.0, 1 - EPSILON)

        diff_norm = gs.sum((point_a - point_b) ** 2, -1)
        norm_function = 1 + 2 * diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))

        return gs.log(norm_function + gs.sqrt(norm_function**2 - 1))

    def retraction(self, tangent_vec, base_point):
        """Poincaré ball model retraction.

        Approximate the exponential map of the Poincare ball

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            vector in tangent space.
        base_point : array-like, shape=[..., dim]
            Second point in the Poincare ball.

        Returns
        -------
        point : array-like, shape=[..., dim]
            Retraction point.

        References
        ----------
        .. [1] Nickel et.al, "Poincaré Embedding for
            Learning Hierarchical Representation", 2017.
        """
        retraction_factor = (
            (1 - gs.sum(base_point**2, axis=-1, keepdims=True)) ** 2
        ) / 4

        return base_point - gs.einsum("...i,...j->...j", retraction_factor, tangent_vec)

    def metric_matrix(self, base_point=None):
        """Compute the inner product matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, defaults to zeros if None.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        if base_point is None:
            base_point = gs.zeros((1, self._space.dim))

        lambda_base = (2 / (1 - gs.sum(base_point * base_point, axis=-1))) ** 2
        identity = gs.eye(self._space.dim, self._space.dim)

        return gs.einsum("...,jk->...jk", lambda_base, identity)

    def normalization_factor(self, variances):
        """Return normalization factor of the Gaussian distribution.

        Parameters
        ----------
        variances : array-like, shape=[n,]
            Array of equally distant values of the
            variance precision time.

        Returns
        -------
        norm_func : array-like, shape=[n,]
            Normalisation factor for all given variances.
        """
        dim = self._space.dim
        binomial_coefficient = None
        n_samples = variances.shape[0]

        expand_variances = gs.expand_dims(variances, axis=0)
        expand_variances = gs.repeat(expand_variances, dim, axis=0)

        if binomial_coefficient is None:
            dim_range = gs.arange(dim)
            dim_range[0] = 1
            n_fact = dim_range.prod()

            k_fact = gs.concatenate(
                [
                    gs.expand_dims(dim_range[:i].prod(), 0)
                    for i in range(1, dim_range.shape[0] + 1)
                ],
                0,
            )

            nmk_fact = gs.flip(k_fact, 0)

            binomial_coefficient = n_fact / (k_fact * nmk_fact)

        binomial_coefficient = gs.expand_dims(binomial_coefficient, -1)
        binomial_coefficient = gs.repeat(binomial_coefficient, n_samples, axis=1)

        range_ = gs.expand_dims(gs.arange(dim), -1)
        range_ = gs.repeat(range_, n_samples, axis=1)

        ones_ = gs.expand_dims(gs.ones(dim), -1)
        ones_ = gs.repeat(ones_, n_samples, axis=1)

        alternate_neg = (-ones_) ** (range_)

        erf_arg = (((dim - 1) - 2 * range_) * expand_variances) / gs.sqrt(2)
        exp_arg = ((((dim - 1) - 2 * range_) * expand_variances) / gs.sqrt(2)) ** 2
        norm_func_1 = (1 + gs.erf(erf_arg)) * gs.exp(exp_arg)
        norm_func_2 = binomial_coefficient * norm_func_1
        norm_func_3 = alternate_neg * norm_func_2

        norm_func = (
            NORMALIZATION_FACTOR_CST
            * variances
            * norm_func_3.sum(0)
            * (1 / (2 ** (dim - 1)))
        )

        return norm_func

    def _compute_alpha(self, current_dim):
        """Compute factor used in normalization factor.

        Compute alpha factor given the two arguments.

        Parameters
        ----------
        current_dim : array-like, shape=[dim,]
            Array initialized at 0 with max range dim and step 1.
        """
        return (self._space.dim - 1 - 2 * current_dim) / SQRT_2

    def norm_factor_gradient(self, variances):
        """Compute normalization factor and its gradient.

        Compute normalization factor given current variance
        and dimensionality.

        Parameters
        ----------
        variances : array-like, shape=[n]
            Value of variance.

        Returns
        -------
        norm_factor : array-like, shape=[n]
            Normalisation factor.
        norm_factor_gradient : array-like, shape=[n]
            Gradient of the normalization factor.
        """
        dim = self._space.dim
        variances = gs.transpose(gs.to_ndarray(variances, to_ndim=2))
        dim_range = gs.arange(0, dim, 1.0)
        alpha = self._compute_alpha(dim_range)

        binomial_coefficient = gs.ones(dim)
        binomial_coefficient[1:] = (dim - 1 + 1 - dim_range[1:]) / dim_range[1:]
        binomial_coefficient = gs.cumprod(binomial_coefficient)

        beta = ((-gs.ones(dim)) ** dim_range) * binomial_coefficient

        sigma_repeated = gs.repeat(variances, dim, -1)
        prod_alpha_sigma = gs.einsum("ij,j->ij", sigma_repeated, alpha)
        term_2 = gs.exp((prod_alpha_sigma) ** 2) * (1 + gs.erf(prod_alpha_sigma))
        term_1 = gs.sqrt(gs.pi / 2.0) * (1.0 / (2 ** (dim - 1)))
        term_2 = gs.einsum("ij,j->ij", term_2, beta)
        norm_factor = term_1 * variances * gs.sum(term_2, axis=-1, keepdims=True)
        grad_term_1 = 1 / variances

        grad_term_21 = 1 / gs.sum(term_2, axis=-1, keepdims=True)

        grad_term_211 = (
            gs.exp((prod_alpha_sigma) ** 2)
            * (1 + gs.erf(prod_alpha_sigma))
            * gs.einsum("ij,j->ij", sigma_repeated, alpha**2)
            * 2
        )

        grad_term_212 = gs.repeat(
            gs.expand_dims((2 / gs.sqrt(gs.pi)) * alpha, axis=0),
            variances.shape[0],
            axis=0,
        )

        grad_term_22 = grad_term_211 + grad_term_212
        grad_term_22 = gs.einsum("ij, j->ij", grad_term_22, beta)
        grad_term_22 = gs.sum(grad_term_22, axis=-1, keepdims=True)

        norm_factor_gradient = grad_term_1 + (grad_term_21 * grad_term_22)

        return gs.squeeze(norm_factor), gs.squeeze(norm_factor_gradient)

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
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
        radius : float
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space, radius, base_point)
