"""Gaussian distributions on manifolds."""
import geomstats.backend as gs

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

        num = gs.exp(-(metric.dist(data_expanded, means_expanded) ** 2) / (2 * variances_expanded ** 2))

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
        variances_units = gs.repeat(variances_units, distance_to_mean.shape[0], axis=0)

        distribution_normal = gs.exp(-(distance_to_mean ** 2) / (2 * variances_units ** 2))

        zeta_sigma = PI_2_3 * variances * gs.exp((variances ** 2 / 2) * gs.erf(variances / gs.sqrt(2)))

        result_num = gs.expand_dims(mixture_coefficients, 0)
        result_num = gs.repeat(result_num, len(distribution_normal), axis=0) * distribution_normal
        result_denum = gs.expand_dims(zeta_sigma, 0)
        result_denum = gs.repeat(result_denum, len(distribution_normal), axis=0)

        result = result_num / result_denum

        return result


class Normalization_Factor_Storage(object):
    """A class for computing the normalization factor."""

    def __init__(self, variances, dimension):

        self.dimension = dimension

        self.variances = variances
        self.m_zeta_var = self.new_zeta(variances, dimension)

        c1 = self.m_zeta_var.sum() != self.m_zeta_var.sum()
        c2 = self.m_zeta_var.sum() == gs.inf
        c3 = self.m_zeta_var.sum() == -gs.inf

        if (c1 or c2 or c3):
            print("WARNING : ZetaPhiStorage , untracktable normalisation factor :")
            max_nf = len(variances)

            limit_nf = ((self.m_zeta_var / self.m_zeta_var) * 0).nonzero()[0].item()
            self.variances = self.variances[0:limit_nf]
            self.m_zeta_var = self.m_zeta_var[0:limit_nf]
            if (c1):
                print("\t Nan value in processing normalisation factor")
            if (c2 or c3):
                print("\t +-inf value in processing normalisation factor")

            print("\t Max variance is now : ", self.variances[-1])
            print("\t Number of possible variance is now : " + str(len(self.variances)) + "/" + str(max_nf))

        factor_normalization, log_grad_zeta = self.zeta_dlogzetat(self.variances, dimension)

        self.phi_inv_var = self.variances ** 3 * log_grad_zeta

    def normalisation_factor(self, variance):
        """Given a variance, finds the normalisation factor."""
        N, P = variance.shape[0], self.variances.shape[0]

        ref = gs.expand_dims(self.variances, 0)
        ref = gs.repeat(ref, N, axis=0)
        val = gs.expand_dims(variance, 1)
        val = gs.repeat(val, P, axis=1)

        difference = gs.abs(ref - val)

        index = gs.argmin(difference, axis=-1)

        return self.m_zeta_var[index]

    def phi(self, phi_val):
        N, P = phi_val.shape[0], self.variances.shape[0]

        ref = gs.expand_dims(self.phi_inv_var, 0)
        ref = gs.repeat(ref, N, axis=0)

        val = gs.expand_dims(phi_val, 1)
        val = gs.repeat(val, P, axis=1)

        abs_difference = gs.abs(ref - val)

        index = gs.argmin(abs_difference, -1)

        return self.variances[index]


    def new_zeta(self,sigma, N):
        binomial_coefficient = None
        M = sigma.shape[0]

        sigma_u = gs.expand_dims(sigma, axis=0)
        sigma_u = gs.repeat(sigma_u, N, axis=0)

        if (binomial_coefficient is None):
            # we compute coeficient
            # v = torch.arange(N)
            v = gs.arange(N)
            v[0] = 1
            n_fact = v.prod()

            k_fact = gs.concatenate([gs.expand_dims(v[:i].prod(), 0) for i in range(1, v.shape[0] + 1)], 0)

            nmk_fact = gs.flip(k_fact, 0)

            binomial_coefficient = n_fact / (k_fact * nmk_fact)

        # TODO: Check Precision for binomial coefficients
        binomial_coefficient = gs.expand_dims(binomial_coefficient, -1)
        binomial_coefficient = gs.repeat(binomial_coefficient, M, axis=1)

        range_ = gs.expand_dims(gs.arange(N), -1)
        range_ = gs.repeat(range_, M, axis=1)

        ones_ = gs.expand_dims(gs.ones(N), -1)
        ones_ = gs.repeat(ones_, M, axis=1)

        alternate_neg = (-ones_) ** (range_)

        ins_gs = (((N - 1) - 2 * range_) * sigma_u) / gs.sqrt(2)
        ins_squared_gs = ((((N - 1) - 2 * range_) * sigma_u) / gs.sqrt(2)) ** 2
        as_o_gs = (1 + gs.erf(ins_gs)) * gs.exp(ins_squared_gs)
        bs_o_gs = binomial_coefficient * as_o_gs
        r_gs = alternate_neg * bs_o_gs

        zeta = NORMALIZATION_FACTOR_CST * sigma * r_gs.sum(0) * (1 / (2 ** (N - 1)))

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
        binomial_coefficient[1:] = (dimension - 1 + 1 - dim_range[1:]) / dim_range[1:]
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
