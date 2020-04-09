"""Expectation maximisation algorithm."""

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean
import math


EM_CONV_RATE = 1e-4
MINIMUM_EPOCHS = 10
DEFAULT_MAX_ITER = 100
DEFAULT_LR = 5e-2
DEFAULT_CONV_FACTOR = 1e-4
DEFAULT_TOL = 1e-2
ZETA_CST = gs.sqrt(gs.pi/2)
PI_2_3 = pow((2 * gs.pi), 2 / 3)
CST_FOR_ERF = 8.0 / (3.0 * gs.pi) * (gs.pi - 3.0) / (4.0 - gs.pi)
SQRT_2 = gs.sqrt(2.)
ZETA_LOWER_BOUND = 5e-2
ZETA_UPPER_BOUND = 2.
ZETA_STEP = 0.001

class RiemannianEM():

    def __init__(self,
                 riemannian_metric,
                 n_gaussian=8,
                 initialisation_method='random',
                 tol=DEFAULT_TOL,
                 mean_method='default',
                 point_type='vector',
                 verbose=0):
        """Expectation-maximization algorithm on hyperbolic space.

        A class for performing Expectation-Maximization on hyperbolic
        space to fit data into a Gaussian Mixture Model.

        Parameters
        ----------
        n_gaussian : int
        Number of Gaussian components in the mix

        riemannian_metric : object of class RiemannianMetric
        The geomstats Riemmanian metric associated with
                            the used manifold

        initialisation_method : basestring
        Choice between initialization method for variances, means and weights
               'random' : will select random uniformally train point as
                         initial centroids

                #TODO: implement kmeans initialisation
                'kmeans' : will apply Riemannian kmeans to deduce
                variances and means that the EM will use initially

        tol : float
            Convergence factor. If the difference of mean distance
            between two step is lower than tol

        verbose : int
            If verbose > 0, information will be printed during learning

        mean_method: basestring
            Specify the method to compute the mean.

        point_type: basestring
            Specify whether to use vector or matrix representation

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_gaussian = n_gaussian
        self.riemannian_metric = riemannian_metric
        self.initialisation_method = initialisation_method
        self.tol = tol
        self.verbose = verbose
        self.mean_method = mean_method
        self.point_type = point_type

    def update_posterior_probabilities(self, posterior_probabilities, g_index=-1):
        """ Posterior probabilities update function"""
        if (g_index > 0):
            self.mixture_coefficients[g_index] = gs.mean(posterior_probabilities[:, g_index])
        else:
            self.mixture_coefficients = gs.mean(posterior_probabilities, 0)

    def update_means(self, data, wik, lr_mu, tau_mu, g_index=-1, max_iter=150):
        """ Means update functions"""

        N, D, M = data.shape + (wik.shape[-1],)

        mean = FrechetMean(
            metric=self.riemannian_metric,
            method=self.mean_method,
            max_iter=150,
            point_type=self.point_type)

        data_gs = gs.expand_dims(data,1)
        data_gs = gs.repeat(data_gs,M,axis = 1)

        if(g_index>0):
            mean.fit(data, weights=wik[:,g_index])
            self.means[g_index] = gs.squeeze(mean.estimate_)

        else:
            mean.fit(data_gs, weights = wik)
            self.means = gs.squeeze(mean.estimate_)


    def update_variances(self, z, wik, g_index=-1):
        """Variances update function"""

        N, D, M = z.shape + (self.means.shape[0],)

        if (g_index > 0):

            dtm  = ((self.riemannian_metric.dist(z,
                                                    gs.repeat(self.means[:,g_index], N, 0)) **2
                        ) * wik[:,g_index].sum()) / wik[:,g_index].sum()

            self.variances[:, g_index] = self.normalization_factor.phi(dtm)
        else:


            z_gs = gs.expand_dims(z, 1)
            z_gs = gs.repeat(z_gs,M,axis = 1)
            means_gs = gs.expand_dims(self.means,0)
            means_gs = gs.repeat(means_gs,N,axis = 0)

            wik_gs = wik
            dtm_gs = ((self.riemannian_metric.dist(z_gs,
                             means_gs) ** 2) * wik_gs).sum(0) / wik_gs.sum(0)

            self.variances = self.normalization_factor.phi(dtm_gs)

    def _expectation(self, data):
        """Compute weights_ik given the data, means and variances"""

        probability_distribution_function = gaussianPDF(data,
                                                        self.means,
                                                        self.variances,
                                                        norm_func=self.normalization_factor.zeta_numpy,
                                                        metric =self.riemannian_metric)


        if (probability_distribution_function.mean() !=
                probability_distribution_function.mean()):
            print("EXPECTATION : pdf contain not a number elements")
            quit()


        multi = gs.repeat(gs.expand_dims(self.mixture_coefficients,0),
                          len(probability_distribution_function),
                          axis = 0)

        p_pdf = probability_distribution_function* \
                   multi

        valid_pdf_condition = gs.amin(gs.sum(p_pdf, -1))

        if (valid_pdf_condition <= 1e-15):

            if (self._verbose):
                print("EXPECTATION : pdf.sum(-1) contain zero for ", gs.sum(gs.sum(p_pdf,-1) <= 1e-15), "items")
            p_pdf[gs.sum(p_pdf,-1) <= 1e-15] = 1

        denum_wik = gs.repeat(gs.sum(p_pdf,-1, keepdims=True),p_pdf.shape[-1], axis = 1)

        wik = p_pdf / denum_wik

        if (gs.mean(wik) != gs.mean(wik)):

            print("EXPECTATION : wik contain not a number elements")
            quit()

        if gs.mean(gs.sum(wik,1)) <= 1-1e-4 and gs.mean(gs.sum(wik,1)) >= 1+1e-4:

            print("EXPECTATION : wik don't sum to 1")
            print(gs.sum(wik,1))
            quit()

        return wik

    def _maximization(self, data, wik, lr_mu, tau_mu, max_iter = math.inf ):
        """Given the weights and data, will update the means and variances."""

        self.update_posterior_probabilities(wik)

        if(gs.mean(self.mixture_coefficients) != gs.mean(self.mixture_coefficients)):
            print("UPDATE : w contain not a number elements")
            quit()

        self.update_means(data, wik, lr_mu=lr_mu, tau_mu=tau_mu, max_iter=max_iter)


        if(self.verbose):
            print(self.means)
        if(self.means.mean() != self.means.mean()):
            print("UPDATE : mu contain not a number elements")
            quit()
        if(self.verbose):
            print("sigma b", self.variances)

        self.update_variances(data, wik)

        if(self.verbose):
            print("sigma ", self.variances)
        if(self.variances.mean() != self.variances.mean()):
            print("UPDATE : sigma contain not a number elements")
            quit()

    def fit(self,
            data,
            max_iter= DEFAULT_MAX_ITER,
            lr_mean=DEFAULT_LR,
            conv_factor_mean=DEFAULT_CONV_FACTOR):
        """Fit a Gaussian mixture model (GMM) given the data.

        Alternates between Expectation and Maximisation steps
        for some number of iterations.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        max_iter : int
            Maximum number of iterations
            
        lr_mean : float
            Learning rate for the mean
            
        conv_factor_mean : float
            Convergence factor for the mean

        Returns
        -------
        self : object
            Return the components of the computed
            Gaussian mixture model: means, variances and mixture_coefficients
        """
        if(self.initialisation_method == 'random'):

            self._dimension = data.shape[-1]
            self.means = (gs.random.rand(self.n_gaussian, self._dimension) - 0.5) / self._dimension
            self.variances = gs.random.rand(self.n_gaussian) / 10 + 0.8
            self.mixture_coefficients = gs.ones(self.n_gaussian) / self.n_gaussian
            posterior_probabilities = gs.ones((data.shape[0],
                                               self.means.shape[0]))
            self.normalization_factor = ZetaPhiStorage(gs.arange(ZETA_LOWER_BOUND,
                                                                 ZETA_UPPER_BOUND,
                                                                 ZETA_STEP),
                                                       self._dimension)

        else:
            print('Initialisation method not yet implemented')

        if (self.verbose):
            print("Number of data samples", data.shape[0])
            print("Dimensions", self._dimension)

        for epoch in range(max_iter):
            old_posterior_probabilities = posterior_probabilities

            posterior_probabilities = self._expectation(data)

            condition = gs.mean(gs.abs(old_posterior_probabilities
                                       - posterior_probabilities))
            if(condition < EM_CONV_RATE and epoch>MINIMUM_EPOCHS):

                print('EM converged in ', epoch, 'iterations')
                return self.means, self.variances, self.mixture_coefficients

            self._maximization(data, posterior_probabilities, lr_mu=lr_mean, tau_mu=conv_factor_mean)

        print('WARNING: EM did not converge')

        return self.means, self.variances, self.mixture_coefficients

    def predict(self, data):

        """Predict for each data point its community.

        Given each Gaussin of the computed GMM, we compute for each data point
        the probability to belong to the Gaussian then takes the maximum probability
        taking into account the weight of the Gaussian to attribute the community to
        the data point.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Return array containing for each point the associated Gaussian community
        """
        #TODO Thomas or Hadi: Write prediction method to
        # label points with the cluster maximising the likelihood
        belongs = None
        return belongs


class ZetaPhiStorage(object):
    """A class for computing the normalization factor."""

    def __init__(self, sigma, dimension):

        self.dimension = dimension

        self.sigma = sigma
        self.m_zeta_var = new_zeta(sigma, dimension)


        c1 = self.m_zeta_var.sum() != self.m_zeta_var.sum()
        c2 = self.m_zeta_var.sum() == math.inf
        c3 = self.m_zeta_var.sum() == -math.inf

        if (c1 or c2 or c3):
            print("WARNING : ZetaPhiStorage , untracktable normalisation factor :")
            max_nf = len(sigma)

            limit_nf = ((self.m_zeta_var / self.m_zeta_var) * 0).nonzero()[0].item()
            self.sigma = self.sigma[0:limit_nf]
            self.m_zeta_var = self.m_zeta_var[0:limit_nf]
            if (c1):
                print("\t Nan value in processing normalisation factor")
            if (c2 or c3):
                print("\t +-inf value in processing normalisation factor")

            print("\t Max variance is now : ", self.sigma[-1])
            print("\t Number of possible variance is now : " + str(len(self.sigma)) + "/" + str(max_nf))


        sigma_cube = self.sigma ** 3

        factor_normalization, log_grad_zeta = zeta_dlogzetat(self.sigma, dimension)

        self.phi_inv_var = sigma_cube * log_grad_zeta


    def zeta_numpy(self, sigma):
        N, P = sigma.shape[0], self.sigma.shape[0]

        sigma_self = self.sigma
        ref = gs.expand_dims(sigma_self,0)
        ref = gs.repeat(ref, N, axis= 0)
        val = gs.expand_dims(sigma, 1)
        val = gs.repeat(val,P, axis = 1)

        difference = gs.abs(ref-val)

        index = gs.argmin(difference,axis = -1)

        return self.m_zeta_var[index]

    def phi(self, phi_val):
        N, P = phi_val.shape[0], self.sigma.shape[0]

        ref = gs.expand_dims(self.phi_inv_var, 0)
        ref = gs.repeat(ref,N, axis = 0)


        val = gs.expand_dims(phi_val,1)
        val = gs.repeat(val, P, axis = 1)

        #val = phi_val.unsqueeze(1).expand(N, P)

        # print("val ", val)

        abs_difference = gs.abs(ref-val)

        #values = abs_difference.min(-1)
        index = gs.argmin(abs_difference, -1)

        #values, index = torch.abs(ref - val).min(-1)
        #values_gs,
        return self.sigma[index]

    def to(self, device):
        self.sigma = self.sigma.to(device)
        self.m_zeta_var = self.m_zeta_var.to(device)
        self.phi_inv_var = self.phi_inv_var.to(device)


# def erf_approx(x):
#     return gs.sign(x)*gs.sqrt(1 - gs.exp(-x * x * (4 / gs.pi + CST_FOR_ERF * x * x) / (1 + CST_FOR_ERF * x * x)))

def new_zeta(sigma, N):

    binomial_coefficient = None
    M = sigma.shape[0]

    sigma_u = gs.expand_dims(sigma, axis=0)
    sigma_u = gs.repeat(sigma_u, N, axis = 0)


    if(binomial_coefficient is None):
         # we compute coeficient
         #v = torch.arange(N)
         v = gs.arange(N)
         v[0] = 1
         n_fact = v.prod()

         k_fact = gs.concatenate([gs.expand_dims(v[:i].prod(),0) for i in range(1, v.shape[0] + 1)], 0)
         #k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
         nmk_fact = gs.flip(k_fact,0)
         # print(nmk_fact)
         binomial_coefficient = n_fact / (k_fact * nmk_fact)

    #TODO: Check Precision for binomial coefficients
    binomial_coefficient = gs.expand_dims(binomial_coefficient,-1)
    binomial_coefficient = gs.repeat(binomial_coefficient,M, axis = 1)

    range_ = gs.expand_dims(gs.arange(N),-1)
    range_ = gs.repeat(range_, M, axis = 1)

    ones_ = gs.expand_dims(gs.ones(N),-1)
    ones_ = gs.repeat(ones_, M, axis = 1)

    alternate_neg = (-ones_)**(range_)

    ins_gs = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
    ins_squared_gs = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
    as_o_gs = (1+gs.erf(ins_gs)) * gs.exp(ins_squared_gs)
    bs_o_gs = binomial_coefficient * as_o_gs
    r_gs = alternate_neg * bs_o_gs

    zeta = ZETA_CST * sigma * r_gs.sum(0) * (1/(2**(N-1)))

    return zeta


def gaussianPDF(data, means, variances, norm_func, metric):
    # norm_func = zeta
    # print(x.shape, mu.shape)

    N, D, M = data.shape + (means.shape[0],)

    x_rd = gs.expand_dims(data,1)
    x_rd = gs.repeat(x_rd, M, axis = 1)

    mu_rd = gs.expand_dims(means,0)
    mu_rd = gs.repeat(mu_rd, N, axis = 0)

    sigma_rd = gs.expand_dims(variances,0)
    sigma_rd = gs.repeat(sigma_rd,N,0)

    num = gs.exp(-((metric.dist(x_rd, mu_rd)**2))/(2*(sigma_rd)**2))




    #mu_rd = means.unsqueeze(0).expand(N, M, D)
    #sigma_rd = variances.unsqueeze(0).expand(N, M)
    # computing numerator
    #num = torch.exp(-((distance(x_rd, mu_rd)**2))/(2*(sigma_rd)**2))
    # print("num mean ",num.mean())
    den = norm_func(variances)
    # print("den mean ",den.mean() )
    # print("sigma",num)
    # print("den ", den)
    # print("pdf max ", (num/den.unsqueeze(0).expand(N, M)).max())

    den_gs = gs.expand_dims(den,0)
    den_gs = gs.repeat(den_gs, N, axis = 0)

    result = num/den_gs

    return result




def compute_alpha(dim, current_dim):
    """Compute factor used in normalisation factor.

    Compute alpha factor given the two arguments.
    """
    return (dim - 1 - 2 * current_dim) / SQRT_2


def zeta_dlogzetat(sigma, dim):
    """Compute normalisation factor and its gradient.

    Compute normalisation factor given current variance
    and dimensionality.

    Parameters
    ----------
    sigma : array-like, shape=[n]
        Value of variance.

    dim : int
        Dimension of the space

    Returns
    -------
    normalization_factor : array-like, shape=[n]
    """
    sigma = gs.transpose(gs.to_ndarray(sigma, to_ndim=2))
    dim_range = gs.arange(0, dim, 1.)
    alpha = compute_alpha(dim, dim_range)

    binomial_coefficient = gs.ones(dim)
    binomial_coefficient[1:] = (dim - 1 + 1 - dim_range[1:]) / dim_range[1:]
    binomial_coefficient = gs.cumprod(binomial_coefficient)


    beta = ((-gs.ones(dim)) ** dim_range) * binomial_coefficient
    
    sigma_repeated = gs.repeat(sigma, dim, -1)
    prod_alpha_sigma = gs.einsum('ij,j->ij', sigma_repeated, alpha)
    term_2 =\
        gs.exp((prod_alpha_sigma)**2) * (1 + gs.erf(prod_alpha_sigma))
    term_1 = math.sqrt(gs.pi / 2.) * (1. / (2**(dim - 1)))
    term_2 = gs.einsum('ij,j->ij', term_2, beta)
    normalisation_coef =\
        term_1 * sigma * gs.sum(term_2, axis=-1, keepdims=True)
    grad_term_1 = 1 / sigma

    grad_term_21 = 1 / gs.sum(term_2, axis=-1, keepdims=True)

    grad_term_211 =\
        gs.exp((prod_alpha_sigma)**2)\
        * (1 + gs.erf(prod_alpha_sigma))\
        * gs.einsum('ij,j->ij', sigma_repeated, alpha**2) * 2

    grad_term_212 = gs.repeat(gs.expand_dims((2 / math.sqrt(gs.pi))
                              * alpha, axis=0),
                              sigma.shape[0], axis=0)

    grad_term_22 = grad_term_211 + grad_term_212
    grad_term_22 = gs.einsum("ij, j->ij", grad_term_22, beta)
    grad_term_22 = gs.sum(grad_term_22, axis=-1, keepdims=True)

    log_grad_zeta = grad_term_1 + (grad_term_21 * grad_term_22)

    return gs.squeeze(normalisation_coef), gs.squeeze(log_grad_zeta)


def weighted_gmm_pdf(mixture_coefficients,
                     mesh_data,
                     means,
                     variances,
                     metric):

    mesh_data_units = gs.expand_dims(mesh_data, 1)

    mesh_data_units = gs.repeat(mesh_data_units, len(means), axis = 1)

    means_units = gs.expand_dims(means,0)

    means_units = gs.repeat(means_units,mesh_data_units.shape[0],axis = 0)

    distance_to_mean = metric(mesh_data_units, means_units)
    variances_units = gs.expand_dims(variances,0)
    variances_units = gs.repeat(variances_units, distance_to_mean.shape[0], axis = 0)

    distribution_normal = gs.exp(-((distance_to_mean)**2)/(2 * variances_units**2))

    zeta_sigma =PI_2_3 * variances * gs.exp((variances ** 2 / 2) * gs.erf(variances / gs.sqrt(2)))

    result_num_gs = gs.expand_dims(mixture_coefficients,0)
    result_num_gs = gs.repeat(result_num_gs,len(distribution_normal), axis = 0) * distribution_normal
    result_denum_gs = gs.expand_dims(zeta_sigma,0)
    result_denum_gs = gs.repeat(result_denum_gs,len(distribution_normal), axis = 0)

    result = result_num_gs/result_denum_gs

    return result
