"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded with
the hyperboloid representation (embedded in minkowsky space).
"""
import logging

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-6
NORMALIZATION_FACTOR_CST = gs.sqrt(gs.pi / 2)
PI_2_3 = gs.power(gs.array([2. * gs.pi]), gs.array([2 / 3]))
SQRT_2 = gs.sqrt(2.)


class PoincareBall(Hyperbolic):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in the Poincaré ball model.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    scale : int
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
        Optional, default: 1.
    """

    default_coords_type = 'ball'
    default_point_type = 'vector'

    def __init__(self, dim, scale=1):
        super(PoincareBall, self).__init__(
            dim=dim,
            scale=scale)
        self.coords_type = PoincareBall.default_coords_type
        self.point_type = PoincareBall.default_point_type
        self.metric = PoincareBallMetric(self.dim, self.scale)

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be tested.
        tolerance : float, optional
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.
            Optional, default: 1e-6.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        return gs.sum(point**2, axis=-1) < (1 - tolerance)

    @staticmethod
    def gmm_pdf(
            data, means, variances, norm_func,
            metric, variances_range, norm_func_var):
        """Return the separate probability density function of GMM.

        The probability density function is computed for
        each component of the GMM separately (i.e., mixture coefficients
        are not taken into account).

        Parameters
        ----------
        data : array-like, shape=[n_samples, dim]
            Points at which the GMM probability density is computed.
        means : array-like, shape=[n_gaussians, dim]
            Means of each component of the GMM.
        variances : array-like, shape=[n_gaussians,]
            Variances of each component of the GMM.
        norm_func : function
            Normalisation factor function.
        metric : function
            Distance function associated with the used metric.

        Returns
        -------
        pdf : array-like, shape=[n_samples, n_gaussians,]
            Probability density function computed at each data
            sample and for each component of the GMM.
        """
        data_length, _, _ = data.shape + (means.shape[0],)

        variances_expanded = gs.expand_dims(variances, 0)
        variances_expanded = gs.repeat(variances_expanded, data_length, 0)

        variances_flatten = variances_expanded.flatten()

        distances = -(metric.dist_broadcast(data, means) ** 2)
        distances = gs.reshape(distances, (data.shape[0] * variances.shape[0]))

        num = gs.exp(
            distances / (2 * variances_flatten ** 2))

        den = norm_func(variances, variances_range, norm_func_var)

        den = gs.expand_dims(den, 0)
        den = gs.repeat(den, data_length, axis=0).flatten()

        pdf = num / den
        pdf = gs.reshape(
            pdf, (data.shape[0], means.shape[0]))

        return pdf

    @staticmethod
    def weighted_gmm_pdf(mixture_coefficients,
                         mesh_data,
                         means,
                         variances,
                         metric):
        """Return the probability density function of a GMM.

        Parameters
        ----------
        mixture_coefficients : array-like, shape=[n_gaussians,]
            Coefficients of the Gaussian mixture model.
        mesh_data : array-like, shape=[n_precision, dim]
            Points at which the GMM probability density is computed.
        means : array-like, shape=[n_gaussians, dim]
            Means of each component of the GMM.
        variances : array-like, shape=[n_gaussians,]
            Variances of each component of the GMM.
        metric : function
            Distance function associated with the used metric.

        Returns
        -------
        weighted_pdf : array-like, shape=[n_precision, n_gaussians,]
            Probability density function computed for each point of
            the mesh data, for each component of the GMM.
        """
        distance_to_mean = metric.dist_broadcast(mesh_data, means)

        variances_units = gs.expand_dims(variances, 0)
        variances_units = gs.repeat(
            variances_units, distance_to_mean.shape[0], axis=0)

        distribution_normal = gs.exp(
            -(distance_to_mean ** 2) / (2 * variances_units ** 2))

        zeta_sigma = PI_2_3 * variances
        zeta_sigma = zeta_sigma * gs.exp(
            (variances ** 2 / 2) * gs.erf(variances / gs.sqrt(2)))

        result_num = gs.expand_dims(mixture_coefficients, 0)
        result_num = gs.repeat(
            result_num, len(distribution_normal), axis=0)
        result_num = result_num * distribution_normal

        result_denum = gs.expand_dims(zeta_sigma, 0)
        result_denum = gs.repeat(
            result_denum, len(distribution_normal), axis=0)

        weighted_pdf = result_num / result_denum

        return weighted_pdf


class PoincareBallMetric(RiemannianMetric):
    """Class that defines operations using a Poincare ball.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    scale : int
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
        Optional, default 1.
    """

    default_point_type = 'vector'
    default_coords_type = 'ball'

    def __init__(self, dim, scale=1):
        super(PoincareBallMetric, self).__init__(
            dim=dim,
            signature=(dim, 0, 0))
        self.coords_type = PoincareBall.default_coords_type
        self.point_type = PoincareBall.default_point_type
        self.scale = scale

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point in hyperbolic space.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in hyperbolic space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        norm_base_point = gs.linalg.norm(base_point, axis=-1)
        norm_tan = gs.linalg.norm(tangent_vec, axis=-1)

        den = 1 - norm_base_point ** 2
        lambda_base_point = 1 / den

        zero_tan = gs.isclose(gs.sum(tangent_vec ** 2, axis=-1), 0.)

        if gs.any(zero_tan):
            norm_tan = gs.assignment(norm_tan, EPSILON, zero_tan)

        direction = gs.einsum('...i,...->...i', tangent_vec, 1 / norm_tan)

        factor = gs.tanh(
            gs.einsum('...,...->...', lambda_base_point, norm_tan))

        exp = self.mobius_add(
            base_point,
            gs.einsum('...i,...->...i', direction, factor))

        if gs.any(zero_tan):
            exp = gs.assignment(
                exp, base_point[zero_tan], zero_tan)

        return exp

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        If point_type = 'poincare' then base_point belongs
        to the Poincare ball and point is a vector in the Euclidean
        space of the same dimension as the ball.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in hyperbolic space.
        base_point : array-like, shape=[..., dim]
            Point in hyperbolic space.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        add_base_point = self.mobius_add(-base_point, point)
        norm_add =\
            gs.expand_dims(gs.linalg.norm(
                           add_base_point, axis=-1), axis=-1)

        norm_base_point =\
            gs.expand_dims(gs.linalg.norm(
                           base_point, axis=-1), axis=-1)

        log = (1 - norm_base_point**2) * gs.arctanh(norm_add)

        mask_0 = gs.isclose(gs.squeeze(norm_add, axis=-1), 0.)
        mask_non0 = ~mask_0
        add_base_point = gs.assignment(
            add_base_point,
            gs.zeros_like(add_base_point[mask_0]),
            mask_0)
        add_base_point = gs.assignment(
            add_base_point,
            add_base_point[mask_non0] / norm_add[mask_non0],
            mask_non0)

        log = gs.einsum(
            '...i,...j->...j', log, add_base_point)
        return log

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def mobius_add(self, point_a, point_b):
        r"""Compute the Mobius addition of two points.

        Mobius addition operation that is a necessary operation
        to compute the log and exp using the 'ball' representation.

        .. math::

            a\oplus b=\frac{(1+2\langle a,b\rangle + ||b||^2)a+
            (1-||a||^2)b}{1+2\langle a,b\rangle + ||a||^2||b||^2}

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point in hyperbolic space.
        point_b : array-like, shape=[..., dim]
            Point in hyperbolic space.

        Returns
        -------
        mobius_add : array-like, shape=[...,]
            Result of the Mobius addition.
        """
        ball_manifold = PoincareBall(self.dim, scale=self.scale)
        point_a_belong = ball_manifold.belongs(point_a)
        point_b_belong = ball_manifold.belongs(point_b)

        if (not gs.all(point_a_belong) or not gs.all(point_b_belong)):
            raise ValueError("Points do not belong to the Poincare ball")

        norm_point_a = gs.sum(point_a ** 2, axis=-1, keepdims=True)
        norm_point_b = gs.sum(point_b ** 2, axis=-1, keepdims=True)

        sum_prod_a_b = gs.einsum('...i,...i->...', point_a, point_b)
        sum_prod_a_b = gs.expand_dims(sum_prod_a_b, axis=-1)

        add_num_1 = 1 + 2 * sum_prod_a_b + norm_point_b
        add_num_1 = gs.einsum('...i,...k->...k', add_num_1, point_a)
        add_num_2 = gs.einsum('...i,...k->...k', (1 - norm_point_a), point_b)
        add_nominator = add_num_1 + add_num_2

        add_denominator = (1 + 2 * sum_prod_a_b + norm_point_a * norm_point_b)

        mobius_add = gs.einsum(
            '...i,...k->...i', add_nominator, 1 / add_denominator)

        return mobius_add

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def dist_broadcast(self, point_a, point_b):
        """Compute the geodesic distance between points.

        If n_samples_a == n_samples_b then dist is the element-wise
        distance result of a point in points_a with the point from
        points_b of the same index. If n_samples_a not equal to
        n_samples_b then dist is the result of applying geodesic
        distance for each point from points_a to all points from
        points_b.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples_a, dim]
            Set of points in hyperbolic space.
        point_b : array-like, shape=[n_samples_b, dim]
            Second set of points in hyperbolic space.

        Returns
        -------
        dist : array-like,
            shape=[n_samples_a, dim] or [n_samples_a, n_samples_b, dim]
            Geodesic distance between the two points.
        """
        if point_a.shape[-1] != point_b.shape[-1]:
            raise ValueError('Manifold dimensions not equal')

        if point_a.shape[0] != point_b.shape[0]:

            point_a_broadcast, point_b_broadcast = gs.broadcast_arrays(
                point_a[:, None], point_b[None, ...])

            point_a_flatten = gs.reshape(
                point_a_broadcast, (-1, point_a_broadcast.shape[-1]))
            point_b_flatten = gs.reshape(
                point_b_broadcast, (-1, point_b_broadcast.shape[-1]))

            point_a_norm = gs.clip(gs.sum(
                point_a_flatten ** 2, -1), 0., 1 - EPSILON)
            point_b_norm = gs.clip(gs.sum(
                point_b_flatten ** 2, -1), 0., 1 - EPSILON)

            square_diff = (point_a_flatten - point_b_flatten) ** 2

            diff_norm = gs.sum(square_diff, -1)
            norm_function = 1 + 2 * \
                diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))

            dist = gs.log(norm_function + gs.sqrt(norm_function ** 2 - 1))
            dist *= self.scale
            dist = gs.reshape(dist, (point_a.shape[0], point_b.shape[0]))
            dist = gs.squeeze(dist)

        elif point_a.shape == point_b.shape:
            dist = self.dist(point_a, point_b)

        return dist

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            First point in hyperbolic space.
        point_b : array-like, shape=[..., dim]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[...,]
            Geodesic distance between the two points.
        """
        point_a_norm = gs.clip(gs.sum(point_a ** 2, -1), 0., 1 - EPSILON)
        point_b_norm = gs.clip(gs.sum(point_b ** 2, -1), 0., 1 - EPSILON)

        diff_norm = gs.sum((point_a - point_b) ** 2, -1)
        norm_function = 1 + 2 * \
            diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))

        dist = gs.log(norm_function + gs.sqrt(norm_function ** 2 - 1))
        dist *= self.scale
        return dist

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def retraction(self, tangent_vec, base_point):
        """Poincaré ball model retraction.

        Approximate the exponential map of hyperbolic space
        .. [1] nickel et.al, "Poincaré Embedding for
         Learning Hierarchical Representation", 2017.


        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            vector in tangent space.
        base_point : array-like, shape=[..., dim]
            Second point in hyperbolic space.

        Returns
        -------
        point : array-like, shape=[..., dim]
            Retraction point.
        """
        ball_manifold = PoincareBall(self.dim, scale=self.scale)
        base_point_belong = ball_manifold.belongs(base_point)

        if not gs.all(base_point_belong):
            raise NameError("Points do not belong to the Poincare ball")

        retraction_factor =\
            ((1 - gs.sum(base_point**2, axis=-1, keepdims=True))**2) / 4

        return base_point\
            - gs.einsum('...i,...j->...j', retraction_factor, tangent_vec)

    @geomstats.vectorization.decorator(['else', 'vector'])
    def inner_product_matrix(self, base_point=None):
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
            base_point = gs.zeros((1, self.dim))

        lambda_base =\
            (2 / (1 - gs.sum(base_point * base_point, axis=-1)))**2
        identity = gs.eye(self.dim, self.dim)

        return gs.einsum('i,jk->ijk', lambda_base, identity)

    def normalization_factor_init(self, variances):
        r"""Set up function for the normalization factor.

        The normalization factor is used to define Gaussian distributions
        on the Poincaré Ball.

        Parameters
        ----------
        variances : array-like, shape=[n_variances,]
            Array of standard deviations.
        normalization_factor_var : array-like, shape=[n_variances,]
            Array of computed normalization factor.
        phi_inv_var : array-like, shape=[n_variances,]
            Array of the computed inverse of a function phi
            whose expression is closed-form
            :math:`\sigma\mapsto \sigma^3 \times \frac{d  }
            {\mathstrut d\sigma}\log \zeta_m(\sigma)'
            where :math:'\sigma' denotes the variance
            and :math:'\zeta' the normalization coefficient
            and :math:'m' the dimension.

        Returns
        -------
        variances : array-like, shape=[n_variances,]
            Array of standard deviations.
        normalization_factor_var : array-like, shape=[n_variances,]
            Array of computed normalization factor.
        phi_inv_var : array-like, shape=[n_variances,]
            Array of the computed inverse of a function phi
            whose expression is closed-form
            :math:`\sigma\mapsto \sigma^3 \times \frac{d  }
            {\mathstrut d\sigma}\log \zeta_m(\sigma)'
            where :math:'\sigma' denotes the variance
            and :math:'\zeta' the normalization coefficient
            and :math:'m' the dimension.
        """
        normalization_factor_var = \
            self.normalization_factor(variances)

        cond_1 = normalization_factor_var.sum() != \
            normalization_factor_var.sum()
        cond_2 = normalization_factor_var.sum() == float('+inf')
        cond_3 = normalization_factor_var.sum() == float('-inf')

        if cond_1 or cond_2 or cond_3:
            logging.warning(
                'Untracktable normalization factor :')

            limit_nf = ((normalization_factor_var /
                         normalization_factor_var)
                        * 0).nonzero()[0].item()
            max_nf = len(variances)
            variances = variances[0:limit_nf]
            normalization_factor_var = \
                normalization_factor_var[0:limit_nf]
            if cond_1:
                logging.warning('\t Nan value '
                                'in processing normalization factor')
            if cond_2 or cond_3:
                raise ValueError('\t +-inf value in '
                                 'processing normalization factor')

            logging.warning('\t Max variance is now : %s',
                            str(variances[-1]))
            logging.warning('\t Number of possible variance is now: %s / %s ',
                            str(len(variances)), str(max_nf))

        _, log_grad_zeta = \
            self.norm_factor_gradient(variances)

        phi_inv_var = variances ** 3 * log_grad_zeta

        return \
            variances, normalization_factor_var, phi_inv_var

    @staticmethod
    def find_normalization_factor(
            variance, variances_range, normalization_factor_var):
        """Find the normalization factor given some variances.

        Parameters
        ----------
        variance : array-like, shape=[n_gaussians,]
            Array of standard deviations for each component
            of some GMM.
        variances_range : array-like, shape=[n_variances,]
            Array of standard deviations.
        normalization_factor_var : array-like, shape=[n_variances,]
            Array of computed normalization factor.

        Returns
        -------
        norm_factor : array-like, shape=[n_gaussians,]
            Array of normalization factors for the given
            variances.
        """
        n_gaussians, precision = variance.shape[0], variances_range.shape[0]

        ref = gs.expand_dims(variances_range, 0)
        ref = gs.repeat(ref, n_gaussians, axis=0)
        val = gs.expand_dims(variance, 1)
        val = gs.repeat(val, precision, axis=1)

        difference = gs.abs(ref - val)

        index = gs.argmin(difference, axis=-1)
        norm_factor = normalization_factor_var[index]

        return norm_factor

    @staticmethod
    def find_variance_from_index(
            weighted_distances, variances_range, phi_inv_var):
        r"""Return the variance given weighted distances.

        Parameters
        ----------
        weighted_distances : array-like, shape=[n_gaussians,]
            Mean of the weighted distances between training data
            and current barycentres. The weights of each data sample
            corresponds to the probability of belonging to a component
            of the Gaussian mixture model.
        variances_range : array-like, shape=[n_variances,]
            Array of standard deviations.
        phi_inv_var : array-like, shape=[n_variances,]
            Array of the computed inverse of a function phi
            whose expression is closed-form
            :math:`\sigma\mapsto \sigma^3 \times \frac{d  }
            {\mathstrut d\sigma}\log \zeta_m(\sigma)'
            where :math:'\sigma' denotes the variance
            and :math:'\zeta' the normalization coefficient
            and :math:'m' the dimension.

        Returns
        -------
        var : array-like, shape=[n_gaussians,]
            Estimated variances for each component of the GMM.
        """
        n_gaussians, precision = \
            weighted_distances.shape[0], variances_range.shape[0]

        ref = gs.expand_dims(phi_inv_var, 0)
        ref = gs.repeat(ref, n_gaussians, axis=0)

        val = gs.expand_dims(weighted_distances, 1)
        val = gs.repeat(val, precision, axis=1)

        abs_difference = gs.abs(ref - val)

        index = gs.argmin(abs_difference, -1)

        var = variances_range[index]

        return var

    def normalization_factor(self, variances):
        """Return normalization factor.

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
        binomial_coefficient = None
        n_samples = variances.shape[0]

        expand_variances = gs.expand_dims(variances, axis=0)
        expand_variances = gs.repeat(expand_variances, self.dim, axis=0)

        if binomial_coefficient is None:

            dim_range = gs.arange(self.dim)
            dim_range[0] = 1
            n_fact = dim_range.prod()

            k_fact = gs.concatenate(
                [gs.expand_dims(dim_range[:i].prod(), 0)
                 for i in range(1, dim_range.shape[0] + 1)], 0)

            nmk_fact = gs.flip(k_fact, 0)

            binomial_coefficient = n_fact / (k_fact * nmk_fact)

        binomial_coefficient = gs.expand_dims(binomial_coefficient, -1)
        binomial_coefficient = gs.repeat(binomial_coefficient,
                                         n_samples, axis=1)

        range_ = gs.expand_dims(gs.arange(self.dim), -1)
        range_ = gs.repeat(range_, n_samples, axis=1)

        ones_ = gs.expand_dims(gs.ones(self.dim), -1)
        ones_ = gs.repeat(ones_, n_samples, axis=1)

        alternate_neg = (-ones_) ** (range_)

        erf_arg = (((self.dim - 1) - 2 * range_) *
                   expand_variances) / gs.sqrt(2)
        exp_arg = ((((self.dim - 1) - 2 * range_) *
                    expand_variances) / gs.sqrt(2)) ** 2
        norm_func_1 = (1 + gs.erf(erf_arg)) * gs.exp(exp_arg)
        norm_func_2 = binomial_coefficient * norm_func_1
        norm_func_3 = alternate_neg * norm_func_2

        norm_func = NORMALIZATION_FACTOR_CST * variances * \
            norm_func_3.sum(0) * (1 / (2 ** (self.dim - 1)))

        return norm_func

    def _compute_alpha(self, current_dim):
        """Compute factor used in normalization factor.

        Compute alpha factor given the two arguments.

        Parameters
        ----------
        current_dim : array-like, shape=[self.dim,]
            Array initialized at 0 with max range self.dim and step 1.
        """
        return (self.dim - 1 - 2 * current_dim) / SQRT_2

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
        variances = gs.transpose(gs.to_ndarray(variances, to_ndim=2))
        dim_range = gs.arange(0, self.dim, 1.)
        alpha = self._compute_alpha(dim_range)

        binomial_coefficient = gs.ones(self.dim)
        binomial_coefficient[1:] = \
            (self.dim - 1 + 1 - dim_range[1:]) / dim_range[1:]
        binomial_coefficient = gs.cumprod(binomial_coefficient)

        beta = ((-gs.ones(self.dim)) ** dim_range) * binomial_coefficient

        sigma_repeated = gs.repeat(variances, self.dim, -1)
        prod_alpha_sigma = gs.einsum('ij,j->ij', sigma_repeated, alpha)
        term_2 = \
            gs.exp((prod_alpha_sigma) ** 2) * (1 + gs.erf(prod_alpha_sigma))
        term_1 = gs.sqrt(gs.pi / 2.) * (1. / (2 ** (self.dim - 1)))
        term_2 = gs.einsum('ij,j->ij', term_2, beta)
        norm_factor = \
            term_1 * variances * gs.sum(term_2, axis=-1, keepdims=True)
        grad_term_1 = 1 / variances

        grad_term_21 = 1 / gs.sum(term_2, axis=-1, keepdims=True)

        grad_term_211 = \
            gs.exp((prod_alpha_sigma) ** 2) \
            * (1 + gs.erf(prod_alpha_sigma)) \
            * gs.einsum('ij,j->ij', sigma_repeated, alpha ** 2) * 2

        grad_term_212 = gs.repeat(gs.expand_dims((2 / gs.sqrt(gs.pi))
                                                 * alpha, axis=0),
                                  variances.shape[0], axis=0)

        grad_term_22 = grad_term_211 + grad_term_212
        grad_term_22 = gs.einsum('ij, j->ij', grad_term_22, beta)
        grad_term_22 = gs.sum(grad_term_22, axis=-1, keepdims=True)

        norm_factor_gradient = grad_term_1 + (grad_term_21 * grad_term_22)

        return gs.squeeze(norm_factor), gs.squeeze(norm_factor_gradient)
