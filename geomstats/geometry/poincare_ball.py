"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded with
the hyperboloid representation (embedded in minkowsky space).
"""
import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-6


class PoincareBall(Hyperbolic):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in the Poincaré ball model.

    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_coords_type = 'ball'
    default_point_type = 'vector'

    def __init__(self, dimension, scale=1):
        super(PoincareBall, self).__init__(
            dimension=dimension,
            scale=scale)
        self.coords_type = PoincareBall.default_coords_type
        self.point_type = PoincareBall.default_point_type
        self.metric =\
            PoincareBallMetric(self.dimension, self.scale)

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Point to be tested.
        tolerance : float, optional
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        return gs.sum(point**2, axis=-1) < (1 - tolerance)


class PoincareBallMetric(RiemannianMetric):
    """Class that defines operations using a Poincare ball.

    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_point_type = 'vector'
    default_coords_type = 'ball'

    def __init__(self, dimension, scale=1):
        super(PoincareBallMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.coords_type = PoincareBall.default_coords_type
        self.point_type = PoincareBall.default_point_type
        self.scale = scale

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension]
            Tangent vector at a base point.
        base_point : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space.

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        norm_base_point = gs.to_ndarray(
            gs.linalg.norm(base_point, axis=-1), 2, axis=-1)
        norm_base_point = gs.to_ndarray(norm_base_point, to_ndim=2)

        norm_base_point = gs.repeat(
            norm_base_point, base_point.shape[-1], axis=-1)
        den = 1 - norm_base_point**2

        norm_tan = gs.to_ndarray(gs.linalg.norm(
            tangent_vec, axis=-1), 2, axis=-1)
        norm_tan = gs.repeat(norm_tan, base_point.shape[-1], -1)

        lambda_base_point = 1 / den

        zero_tan = gs.isclose((tangent_vec * tangent_vec).sum(axis=-1), 0.)

        if norm_tan[zero_tan].shape[0] != 0:
            norm_tan[zero_tan] = EPSILON

        direction = tangent_vec / norm_tan

        factor = gs.tanh(lambda_base_point * norm_tan)

        exp = self.mobius_add(base_point, direction * factor)

        zero_tan = gs.isclose((tangent_vec * tangent_vec).sum(axis=-1), 0.)

        if exp[zero_tan].shape[0] != 0:
            exp[zero_tan] = base_point[zero_tan]

        return exp

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        If point_type = 'poincare' then base_point belongs
        to the Poincare ball and point is a vector in the Euclidean
        space of the same dimension as the ball.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space.
        base_point : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space.

        Returns
        -------
        log : array-like, shape=[n_samples, dimension]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        add_base_point = self.mobius_add(-base_point, point)
        norm_add = gs.to_ndarray(gs.linalg.norm(
            add_base_point, axis=-1), 2, -1)
        norm_add = gs.repeat(norm_add, base_point.shape[-1], -1)
        norm_base_point = gs.to_ndarray(gs.linalg.norm(
            base_point, axis=-1), 2, -1)
        norm_base_point = gs.repeat(norm_base_point,
                                    base_point.shape[-1], -1)

        log = (1 - norm_base_point**2) * gs.arctanh(norm_add)\
            * (add_base_point / norm_add)

        mask_0 = gs.isclose(norm_add, 0.)
        log[mask_0] = 0

        return log

    def mobius_add(self, point_a, point_b):
        r"""Compute the Mobius addition of two points.

        Mobius addition operation that is a necessary operation
        to compute the log and exp using the 'ball' representation.

        .. math::

            a\oplus b=\frac{(1+2\langle a,b\rangle + ||b||^2)a+
            (1-||a||^2)b}{1+2\langle a,b\rangle + ||a||^2||b||^2}

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space.
        point_b : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space.

        Returns
        -------
        mobius_add : array-like, shape=[n_samples, 1]
            Result of the Mobius addition.
        """
        ball_manifold = PoincareBall(self.dimension, scale=self.scale)
        point_a_belong = ball_manifold.belongs(point_a)
        point_b_belong = ball_manifold.belongs(point_b)

        if(not gs.all(point_a_belong) or not gs.all(point_b_belong)):
            raise NameError("Point do not belong to the Poincare ball")

        norm_point_a = gs.sum(point_a ** 2, axis=-1,
                              keepdims=True)

        norm_point_a = gs.repeat(norm_point_a, point_a.shape[-1], -1)

        norm_point_b = gs.sum(point_b ** 2, axis=-1,
                              keepdims=True)
        norm_point_b = gs.repeat(norm_point_b, point_a.shape[-1], -1)

        sum_prod_a_b = gs.sum(point_a * point_b,
                              axis=-1, keepdims=True)

        sum_prod_a_b = gs.repeat(sum_prod_a_b, point_a.shape[-1], -1)

        add_nominator = ((1 + 2 * sum_prod_a_b + norm_point_b) * point_a +
                         (1 - norm_point_a) * point_b)

        add_denominator = (1 + 2 * sum_prod_a_b + norm_point_a * norm_point_b)

        mobius_add = add_nominator / add_denominator

        return mobius_add

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension]
            First point in hyperbolic space.
        point_b : array-like, shape=[n_samples, dimension]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
            Geodesic distance between the two points.
        """
        point_a_norm = gs.clip(gs.sum(point_a ** 2, -1), 0., 1 - EPSILON)
        point_b_norm = gs.clip(gs.sum(point_b ** 2, -1), 0., 1 - EPSILON)

        diff_norm = gs.sum((point_a - point_b) ** 2, -1)
        norm_function = 1 + 2 * \
            diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))

        dist = gs.log(norm_function + gs.sqrt(norm_function ** 2 - 1))
        dist = gs.to_ndarray(dist, to_ndim=1)
        dist = gs.to_ndarray(dist, to_ndim=2, axis=1)
        dist *= self.scale
        return dist

    def retraction(self, tangent_vec, base_point):
        """Poincaré ball model retraction.

        Approximate the exponential map of hyperbolic space
        .. [1] nickel et.al, "Poincaré Embedding for
         Learning Hierarchical Representation", 2017.


        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension]
            vector in tangent space.
        base_point : array-like, shape=[n_samples, dimension]
            Second point in hyperbolic space.

        Returns
        -------
        point : array-like, shape=[n_samples, dimension]
            Retraction point.
        """
        ball_manifold = PoincareBall(self.dimension, scale=self.scale)
        base_point_belong = ball_manifold.belongs(base_point)

        if not gs.all(base_point_belong):
            raise NameError("Point do not belong to the Poincare ball")

        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        retraction_factor = ((1 - (base_point**2).sum(axis=-1))**2) / 4
        retraction_factor =\
            gs.repeat(gs.expand_dims(retraction_factor, -1),
                      base_point.shape[1],
                      axis=1)
        return base_point - retraction_factor * tangent_vec

    def inner_product_matrix(self, base_point=None):
        """Compute the inner product matrix, independent of the base point.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]

        Returns
        -------
        inner_prod_mat: array-like, shape=[n_samples, dimension, dimension]
        """
        if base_point is None:
            base_point = gs.zeros((1, self.dimension))
        dim = base_point.shape[-1]
        n_sample = base_point.shape[0]

        lambda_base =\
            (2 / (1 - gs.sum(base_point * base_point, axis=-1)))**2

        expanded_lambda_base =\
            gs.expand_dims(gs.expand_dims(lambda_base, axis=-1), -1)
        reshaped_lambda_base =\
            gs.repeat(gs.repeat(expanded_lambda_base, dim, axis=-2),
                      dim, axis=-1)

        identity = gs.eye(self.dimension, self.dimension)
        reshaped_identity =\
            gs.repeat(gs.expand_dims(identity, 0), n_sample, axis=0)

        results = reshaped_lambda_base * reshaped_identity
        return results


NORMALIZATION_FACTOR_CST = gs.sqrt(gs.pi / 2)
PI_2_3 = pow((2 * gs.pi), 2 / 3)
SQRT_2 = gs.sqrt(2.)


class GaussianDistribution():
    """A class for Gaussian distributions."""

    @staticmethod
    def gaussian_pdf(data, means, variances, norm_func, metric):
        """Return the probability density function."""
        data_length, dimension, n_gaussian = data.shape + (means.shape[0],)

        data_expanded = gs.expand_dims(data, 1)
        data_expanded = gs.repeat(data_expanded, n_gaussian, axis=1)

        means_expanded = gs.expand_dims(means, 0)
        means_expanded = gs.repeat(means_expanded, data_length, axis=0)

        variances_expanded = gs.expand_dims(variances, 0)
        variances_expanded = gs.repeat(variances_expanded, data_length, 0)

        num = gs.exp(-(metric.dist(data_expanded, means_expanded) ** 2)
                     / (2 * variances_expanded ** 2))

        den = norm_func(variances)

        den = gs.expand_dims(den, 0)
        den = gs.repeat(den, data_length, axis=0)

        result = num / den

        return result

    @staticmethod
    def weighted_gmm_pdf(mixture_coefficients,
                         mesh_data,
                         means,
                         variances,
                         metric):
        """Return the probability density function of a GMM."""
        mesh_data_units = gs.expand_dims(mesh_data, 1)

        mesh_data_units = gs.repeat(mesh_data_units, len(means), axis=1)

        means_units = gs.expand_dims(means, 0)

        means_units = gs.repeat(means_units, mesh_data_units.shape[0], axis=0)

        distance_to_mean = metric(mesh_data_units, means_units)
        variances_units = gs.expand_dims(variances, 0)
        variances_units = gs.repeat(variances_units,
                                    distance_to_mean.shape[0], axis=0)

        distribution_normal = gs.exp(-(distance_to_mean ** 2)
                                     / (2 * variances_units ** 2))

        zeta_sigma = PI_2_3 * variances
        zeta_sigma = zeta_sigma * gs.exp((variances ** 2 / 2)
                                         * gs.erf(variances / gs.sqrt(2)))

        result_num = gs.expand_dims(mixture_coefficients, 0)
        result_num = gs.repeat(result_num,
                               len(distribution_normal), axis=0)
        result_num = result_num * distribution_normal
        result_denum = gs.expand_dims(zeta_sigma, 0)
        result_denum = gs.repeat(result_denum,
                                 len(distribution_normal), axis=0)

        result = result_num / result_denum

        return result


class Normalization_Factor_Storage(object):
    """A class for computing the normalization factor."""

    def __init__(self, variances, dimension):

        self.dimension = dimension

        self.variances = variances
        self.normalisation_factor_var = \
            self.normalization_factor(variances, dimension)

        cond_1 = self.normalisation_factor_var.sum() != \
            self.normalisation_factor_var.sum()
        cond_2 = self.normalisation_factor_var.sum() == gs.inf
        cond_3 = self.normalisation_factor_var.sum() == -gs.inf

        if (cond_1 or cond_2 or cond_3):
            print('WARNING : ZetaPhiStorage , '
                  'untracktable normalisation factor :')
            max_nf = len(variances)

            limit_nf = ((self.normalisation_factor_var /
                         self.normalisation_factor_var)
                        * 0).nonzero()[0].item()
            self.variances = self.variances[0:limit_nf]
            self.normalisation_factor_var = \
                self.normalisation_factor_var[0:limit_nf]
            if (cond_1):
                print('\t Nan value in processing normalisation factor')
            if (cond_2 or cond_3):
                print('\t +-inf value in processing normalisation factor')

            print('\t Max variance is now : ', self.variances[-1])
            print('\t Number of possible variance is now: '
                  + str(len(self.variances)) + '/' + str(max_nf))

        factor_normalization, log_grad_zeta = \
            self.zeta_dlogzetat(self.variances, dimension)

        self.phi_inv_var = self.variances ** 3 * log_grad_zeta

    def find_normalisation_factor(self, variance):
        """Given a variance, finds the normalisation factor."""
        N, P = variance.shape[0], self.variances.shape[0]

        ref = gs.expand_dims(self.variances, 0)
        ref = gs.repeat(ref, N, axis=0)
        val = gs.expand_dims(variance, 1)
        val = gs.repeat(val, P, axis=1)

        difference = gs.abs(ref - val)

        index = gs.argmin(difference, axis=-1)

        return self.normalisation_factor_var[index]

    def _variance_update_sub_function(self, phi_val):
        N, P = phi_val.shape[0], self.variances.shape[0]

        ref = gs.expand_dims(self.phi_inv_var, 0)
        ref = gs.repeat(ref, N, axis=0)

        val = gs.expand_dims(phi_val, 1)
        val = gs.repeat(val, P, axis=1)

        abs_difference = gs.abs(ref - val)

        index = gs.argmin(abs_difference, -1)

        return self.variances[index]

    def normalization_factor(self, variances, dimension):
        """Return normalization factor."""
        binomial_coefficient = None
        n_samples = variances.shape[0]

        expand_variances = gs.expand_dims(variances, axis=0)
        expand_variances = gs.repeat(expand_variances, dimension, axis=0)

        if (binomial_coefficient is None):

            v = gs.arange(dimension)
            v[0] = 1
            n_fact = v.prod()

            k_fact = gs.concatenate([gs.expand_dims(v[:i].prod(), 0)
                                     for i in range(1, v.shape[0] + 1)], 0)

            nmk_fact = gs.flip(k_fact, 0)

            binomial_coefficient = n_fact / (k_fact * nmk_fact)

        binomial_coefficient = gs.expand_dims(binomial_coefficient, -1)
        binomial_coefficient = gs.repeat(binomial_coefficient,
                                         n_samples, axis=1)

        range_ = gs.expand_dims(gs.arange(dimension), -1)
        range_ = gs.repeat(range_, n_samples, axis=1)

        ones_ = gs.expand_dims(gs.ones(dimension), -1)
        ones_ = gs.repeat(ones_, n_samples, axis=1)

        alternate_neg = (-ones_) ** (range_)

        ins_gs = (((dimension - 1) - 2 * range_) *
                  expand_variances) / gs.sqrt(2)
        ins_squared_gs = ((((dimension - 1) - 2 * range_) *
                           expand_variances) / gs.sqrt(2)) ** 2
        as_o_gs = (1 + gs.erf(ins_gs)) * gs.exp(ins_squared_gs)
        bs_o_gs = binomial_coefficient * as_o_gs
        r_gs = alternate_neg * bs_o_gs

        zeta = NORMALIZATION_FACTOR_CST * variances * \
            r_gs.sum(0) * (1 / (2 ** (dimension - 1)))

        return zeta

    def _compute_alpha(self, dim, current_dim):
        """Compute factor used in normalisation factor.

        Compute alpha factor given the two arguments.
        """
        return (dim - 1 - 2 * current_dim) / SQRT_2

    def zeta_dlogzetat(self, variances, dimension):
        """Compute normalisation factor and its gradient.

        Compute normalisation factor given current variance
        and dimensionality.

        Parameters
        ----------
        variances : array-like, shape=[n]
            Value of variance.

        dimension : int
            Dimension of the space

        Returns
        -------
        normalization_factor : array-like, shape=[n]
        """
        variances = gs.transpose(gs.to_ndarray(variances, to_ndim=2))
        dim_range = gs.arange(0, dimension, 1.)
        alpha = self._compute_alpha(dimension, dim_range)

        binomial_coefficient = gs.ones(dimension)
        binomial_coefficient[1:] = \
            (dimension - 1 + 1 - dim_range[1:]) / dim_range[1:]
        binomial_coefficient = gs.cumprod(binomial_coefficient)

        beta = ((-gs.ones(dimension)) ** dim_range) * binomial_coefficient

        sigma_repeated = gs.repeat(variances, dimension, -1)
        prod_alpha_sigma = gs.einsum('ij,j->ij', sigma_repeated, alpha)
        term_2 = \
            gs.exp((prod_alpha_sigma) ** 2) * (1 + gs.erf(prod_alpha_sigma))
        term_1 = gs.sqrt(gs.pi / 2.) * (1. / (2 ** (dimension - 1)))
        term_2 = gs.einsum('ij,j->ij', term_2, beta)
        normalisation_coef = \
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
        grad_term_22 = gs.einsum("ij, j->ij", grad_term_22, beta)
        grad_term_22 = gs.sum(grad_term_22, axis=-1, keepdims=True)

        log_grad_zeta = grad_term_1 + (grad_term_21 * grad_term_22)

        return gs.squeeze(normalisation_coef), gs.squeeze(log_grad_zeta)
