"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space with Poincare ball model.

Lead author: Hadi Zaatiti.
"""

import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out

EPSILON = 1e-6
NORMALIZATION_FACTOR_CST = gs.sqrt(gs.pi / 2)
PI_2_3 = gs.power(gs.array([2.0 * gs.pi]), gs.array([2 / 3]))
SQRT_2 = gs.sqrt(2.0)


def _squared_norm(vec):
    """Compute the squared Euclidean norm of vectors.

    Parameters
    ----------
    vec : array-like, shape=[..., dim]

    Returns
    -------
    squared_norm : array-like, shape=[...,]
    """
    return gs.sum(vec**2, axis=-1)


def _norm(vec):
    """Compute the Euclidean norm of vectors.

    Parameters
    ----------
    vec : array-like, shape=[..., dim]

    Returns
    -------
    norm : array-like, shape=[...,]
    """
    return gs.linalg.norm(vec, axis=-1)


def _scalarvecmul(scalar, vec):
    """Vectorized scalar vector multiplication.

    Parameters
    ----------
    scalar : array-like, shape=[...]
    vec : array-like, shape=[..., dim]
    """
    return gs.einsum("...,...i->...i", scalar, vec)


def _scalarmatmul(scalar, mat):
    """Vectorized scalar matrix multiplication.

    Parameters
    ----------
    scalar : array-like, shape=[...]
    mat : array-like, shape=[n, m]
    """
    return gs.einsum("...,ij->...ij", scalar, mat)


class PoincareBall(_Hyperbolic, VectorSpaceOpenSet):
    """Class for the n-dimensional Poincare ball.

    Class for the n-dimensional Poincaré ball model. For other
    representations of hyperbolic spaces see the `Hyperbolic` class.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    """

    def __init__(self, dim, equip=True):
        self.coords_type = "ball"
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim),
            intrinsic=True,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PoincareBallMetric

    def belongs(self, point, atol=gs.atol):
        r"""Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation, i.e. if :math:`||x||^2 < 1`.

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
        return _squared_norm(point) < (1 - atol)

    def projection(self, point):
        r"""Project a point on the ball.

        Project a point by clipping such that l2
        norm being lower than 1

        .. math::

            x \mapsto \begin{cases}
            x & \text{if } ||x|| < 1 - \epsilon \\
            (1 - \epsilon) \frac{x}{||x||} & \text{otherwise}
            \end{cases}

        where :math:`\epsilon` is the backend absolute tolerance.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point projected on the ball.
        """
        l2_norm = _norm(point)
        if gs.any(l2_norm >= 1 - gs.atol):
            projected_point = _scalarvecmul(1.0 / l2_norm, point * (1 - gs.atol))
            return -gs.maximum(-projected_point, -point)

        return gs.copy(point)


class PoincareBallMetric(RiemannianMetric):
    """Class that defines operations using a Poincare ball."""

    def exp(self, tangent_vec, base_point):
        r"""Compute the Riemannian exponential of a tangent vector.

        Denoting `base_point` by :math:`x` and `tangent_vec` by :math:`v`,

        .. math::

            \exp_x(v) = x \oplus \tanh\left(\frac{\lambda_x ||v||}{2}\right)
            \frac{v}{||v||}

        where :math:`\lambda_x = 2 / (1 - ||x||^2)` is the conformal factor
        and :math:`\oplus` is the Mobius addition.

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
        norm_tangent_vec = _norm(tangent_vec)
        half_conformal_factor = 1 / (1 - _squared_norm(base_point))

        # avoids dividing by 0
        norm_tangent_vec_eps = gs.where(
            gs.isclose(norm_tangent_vec, 0.0), EPSILON, norm_tangent_vec
        )
        direction = _scalarvecmul(1 / norm_tangent_vec_eps, tangent_vec)

        factor = gs.tanh(half_conformal_factor * norm_tangent_vec)
        return self.mobius_add(base_point, _scalarvecmul(factor, direction))

    def log(self, point, base_point):
        r"""Compute Riemannian logarithm of a point wrt a base point.

        Denoting `base_point` by :math:`x` and `point` by :math:`y`,

        .. math::

            \log_x(y) = (1 - ||x||^2) \, \mathrm{artanh}(||-x \oplus y||)
            \frac{-x \oplus y}{||-x \oplus y||}

        where :math:`\oplus` is the Mobius addition. The factor
        :math:`1 - ||x||^2` is :math:`2 / \lambda_x`, with :math:`\lambda_x`
        the conformal factor.

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
        squared_norm_add = _squared_norm(mobius_addition)
        squared_norm_bp = _squared_norm(base_point)
        coef = (1 - squared_norm_bp) * utils.taylor_exp_even_func(
            squared_norm_add, utils.arctanh_card_close_0
        )
        return _scalarvecmul(coef, mobius_addition)

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
        squared_norm_a = _squared_norm(point_a)
        squared_norm_b = _squared_norm(point_b)
        num_1 = _scalarvecmul(1 + 2 * inner + squared_norm_b, point_a)
        num_2 = _scalarvecmul(1 - squared_norm_a, point_b)
        result = _scalarvecmul(
            1.0 / (1 + 2 * inner + squared_norm_b * squared_norm_a),
            num_1 + num_2,
        )
        return self._space.projection(result)

    @staticmethod
    def _gyration(point_a, point_b, vector):
        r"""Compute the gyration of a vector.

        The gyration :math:`\mathrm{gyr}[a, b]` measures the lack of
        associativity of the Mobius addition and is the linear map
        satisfying

        .. math::

            a \oplus (b \oplus w) = (a \oplus b) \oplus \mathrm{gyr}[a, b] w

        Its closed form is

        .. math::

            \mathrm{gyr}[a, b] w = w + 2 \frac{A a + B b}{D}

        with

        .. math::

            A = -\langle a, w\rangle ||b||^2 + \langle b, w\rangle
            + 2 \langle a, b\rangle \langle b, w\rangle, \quad
            B = -\langle b, w\rangle ||a||^2 - \langle a, w\rangle, \quad
            D = 1 + 2 \langle a, b\rangle + ||a||^2 ||b||^2

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point in the Poincare ball.
        point_b : array-like, shape=[..., dim]
            Point in the Poincare ball.
        vector : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        gyrated_vector : array-like, shape=[..., dim]
            Gyration of the vector.

        References
        ----------
        .. [U2008] Ungar, A. A. "A gyrovector space approach to hyperbolic
            geometry." Synthesis Lectures on Mathematics and Statistics 1.1
            (2008): 1-194.
        """
        squared_norm_a = _squared_norm(point_a)
        squared_norm_b = _squared_norm(point_b)
        inner_ab = gs.sum(point_a * point_b, axis=-1)
        inner_aw = gs.sum(point_a * vector, axis=-1)
        inner_bw = gs.sum(point_b * vector, axis=-1)

        coef_a = -inner_aw * squared_norm_b + inner_bw + 2 * inner_ab * inner_bw
        coef_b = -inner_bw * squared_norm_a - inner_aw
        denominator = 1 + 2 * inner_ab + squared_norm_a * squared_norm_b

        return vector + 2 * (
            _scalarvecmul(coef_a / denominator, point_a)
            + _scalarvecmul(coef_b / denominator, point_b)
        )

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`.

        Denoting `base_point` by :math:`x` and `end_point` by :math:`y`,

        .. math::

            P_{x \rightarrow y}(v) = \frac{\lambda_x}{\lambda_y}
            \mathrm{gyr}[y, -x] v

        where :math:`\lambda_x = 2 / (1 - ||x||^2)` is the conformal factor
        and :math:`\mathrm{gyr}` is the gyration of the Mobius addition.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim]
            Point in the Poincare ball.
        direction : array-like, shape=[..., dim]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., dim]
            Point in the Poincare ball to transport to. Unused if `direction`
            is given.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., dim]
            Transported tangent vector at `exp_(base_point)(direction)`.

        References
        ----------
        .. [GBH2018] Ganea, O., Bécigneul, G., Hofmann, T. "Hyperbolic neural
            networks." Advances in Neural Information Processing Systems 31
            (2018).
        """
        if direction is not None:
            end_point = self.exp(direction, base_point)
        elif end_point is None:
            raise ValueError(
                "Either an end_point or a direction must be given to define the"
                " geodesic along which to transport."
            )

        coef = (1 - _squared_norm(end_point)) / (1 - _squared_norm(base_point))
        gyrated_vec = self._gyration(end_point, -base_point, tangent_vec)
        return _scalarvecmul(coef, gyrated_vec)

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

            d(a, b) = \mathrm{arccosh}\left(1 + 2 \frac{||a - b||^2}
            {(1 - ||a||^2)(1 - ||b||^2)}\right)

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
        point_a_norm = gs.clip(_squared_norm(point_a), 0.0, 1 - EPSILON)
        point_b_norm = gs.clip(_squared_norm(point_b), 0.0, 1 - EPSILON)

        diff_norm = _squared_norm(point_a - point_b)
        norm_function = 1 + 2 * diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))

        return gs.log(norm_function + gs.sqrt(norm_function**2 - 1))

    def retraction(self, tangent_vec, base_point):
        r"""Poincaré ball model retraction.

        Approximate the exponential map of the Poincare ball

        .. math::

            R_x(v) = x - \frac{(1 - ||x||^2)^2}{4} v

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
        retraction_factor = ((1 - _squared_norm(base_point)) ** 2) / 4

        return base_point - _scalarvecmul(retraction_factor, tangent_vec)

    def metric_matrix(self, base_point=None):
        r"""Compute the inner product matrix.

        .. math::

            G(x) = \lambda_x^2 I, \quad \lambda_x = \frac{2}{1 - ||x||^2}

        where :math:`\lambda_x` is the conformal factor.

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

        lambda_base = (2 / (1 - _squared_norm(base_point))) ** 2
        identity = gs.eye(self._space.dim, self._space.dim)

        return _scalarmatmul(lambda_base, identity)

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

    def injectivity_radius(self, base_point=None):
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
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)
