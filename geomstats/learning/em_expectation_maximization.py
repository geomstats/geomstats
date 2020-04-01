"""Expectation maximisation algorithm."""

from random import randint

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
import torch
import math

PI_2_3 = pow((2*math.pi),2/3)
A_FOR_ERF = 8.0/(3.0*gs.pi)*(gs.pi-3.0)/(4.0-gs.pi)
ZETA_CST = math.sqrt(math.pi/2)

EM_CONV_RATE = 1e-4
DEFAULT_MAX_ITER = 100
DEFAULT_LR = 5e-2
DEFAULT_TAU = 1e-4

class RiemannianEM(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self,
                 riemannian_metric,
                 n_gaussian=8,
                 init='random',
                 tol=1e-2,
                 mean_method='default',
                 verbose=0):
        """Expectation-maximization algorithm on Riemannian manifolds.

        Perform expectation-maximization to fit data into a Gaussian
        mixture model.

        Parameters
        ----------
        n_gaussian : int
        Number of Gaussian components in the mix

        riemannian_metric : object of class RiemannianMetric
        The geomstats riemmanian metric associated with
                            the used manifold

        init : string
        Choice between initialization method for variances, means and weights
               'random' : will select random uniformally train point as
                         initial centroids

                'kmeans' : will apply Riemannian kmeans to deduce
                variances and means that the EM will use initially

        tol : convergence factor. If the difference of mean distance
             between two step is lower than tol

        verbose : int
        If verbose > 0, information will be printed during learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_gaussian = n_gaussian
        self.init = init
        self.riemannian_metric = riemannian_metric
        self.tol = tol
        self.verbose = verbose
        self.mean_method = mean_method

    def update_weights(self,X, weights_ik, g_index=-1):
        """ Weights update function
        """

    def update_means(self, X, weights_ik, lr_mu, tau_mu, g_index=-1, max_iter=150):
        """ Means update functions"""

    def update_variances(self, X, weights_ik, g_index=-1):
        """Variances update function"""

    def _expectation(self, data):
        """Compute weights_ik given the data, means and variances"""

        probability_distribution_function =
        wik = gs.ones((data.shape[0], self._mu.shape[0]))
        return wik

    def _maximization(self, data, weights, lr_mu, tau_mu, max_iter = math.inf ):
        """Given the weights and data, will update the means and variances."""

    def fit(self, data, max_iter= DEFAULT_MAX_ITER,
            lr_mu=DEFAULT_LR,
            tau_mu=DEFAULT_TAU):
        """Fit a Gaussian mixture model (GMM) given the data.

        Alternates between Expectation and Maximisation steps
        for some number of iterations.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        max_iter : Maximum number of iterations

        Returns
        -------
        self : object
            Return Gaussian mixture model
        """

        #Initialization

        if(self.init =='random'):

            self._dimension = data.shape[1]
            self._mu = (gs.random.rand(self.n_gaussian, self._dimension) - 0.5) / self._dimension
            self._sigma = gs.random.rand(self.n_gaussian) / 10 + 0.8
            self._w = gs.ones(self.n_gaussian) / self.n_gaussian
            wik = gs.ones((data.shape[0], self._mu.shape[0]))
            #self.zeta_phi = ZetaPhiStorage(gs.arange(5e-2, 2., 0.001), self._dimension)

        if (self.verbose):
            print("Number of data samples", data.shape[0])
            print("Number of Gaussian distribution", self._dimension)
            print("Initial Variances", self._sigma)
            print("Initial Mixture Weights", self._w)
            print("Initial Weights", wik)

        for epoch in range(max_iter):
            old_wik = wik

            wik = self._expectation(data)

            if gs.mean(gs.abs(old_wik-wik)) < EM_CONV_RATE and epoch>10:
                print('EM converged in ', epoch, 'iterations')
                return self._mu, self._sigma, self._w

            self._maximization(data, wik, lr_mu=lr_mu, tau_mu=tau_mu)

        print('WARNING: EM did not converge')

        return self._mu, self._sigma, self._w

    def predict(self, X):

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

        self.phi_inv_var = (self.sigma ** 3 * log_grad_zeta(self.sigma, dimension)).detach()
        print(self.phi_inv_var.type())

    def zeta(self, sigma):
        N, P = sigma.shape[0], self.sigma.shape[0]
        ref = self.sigma.unsqueeze(0).expand(N, P)
        val = sigma.unsqueeze(1).expand(N, P)
        values, index = torch.abs(ref - val).min(-1)

        return self.m_zeta_var[index]

    def phi(self, phi_val):
        N, P = phi_val.shape[0], self.sigma.shape[0]
        ref = self.phi_inv_var.unsqueeze(0).expand(N, P)
        val = phi_val.unsqueeze(1).expand(N, P)
        # print("val ", val)
        values, index = torch.abs(ref - val).min(-1)
        return self.sigma[index]

    def to(self, device):
        self.sigma = self.sigma.to(device)
        self.m_zeta_var = self.m_zeta_var.to(device)
        self.phi_inv_var = self.phi_inv_var.to(device)


def new_zeta(x, N):
    sigma = x
    # print(sigma.grad)
    binomial_coefficient=None
    M = sigma.shape[0]

    sigma_u = gs.expand_dims(sigma,0)

    #sigma_u = gs.expand_dims(sigma_u, )

    #sigma_u = sigma.unsqueeze(0).expand(N,M)

    # if(binomial_coefficient is None):
    #     # we compute coeficient
    #     v = torch.arange(N)
    #     v[0] = 1
    #     n_fact = v.prod()
    #     k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
    #     nmk_fact = k_fact.flip(0)
    #     # print(nmk_fact)
    #     binomial_coefficient = n_fact/(k_fact * nmk_fact)
    # binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).double()
    #
    # range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()
    # ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()
    #
    # alternate_neg = (-ones_)**(range_)
    #
    # ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
    # ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
    # as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
    # bs_o = binomial_coefficient * as_o
    # r = alternate_neg * bs_o
    #
    #
    #zeta = ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1)))
    zeta = ZETA_CST * sigma * (1 / (2 ** (N - 1)))
    return zeta

def log_grad_zeta(x, N):
    #sigma = nn.Parameter(x)
    sigma = x
    # print(sigma.grad)
    binomial_coefficient=None
    M = sigma.shape[0]
    #sigma_u = sigma.unsqueeze(0).expand(N,M)

    #TODO@HADI: CHECK EXPAND
    sigma_u = gs.expand_dims(sigma,0)

    if(binomial_coefficient is None):
        # we compute coeficient
        v = torch.arange(N)
        v[0] = 1
        n_fact = v.prod()
        k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
        nmk_fact = k_fact.flip(0)
        binomial_coefficient = n_fact/(k_fact * nmk_fact)
    binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).double()
    #range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()
    range_ = gs.arange(N).unsqueeze(-1).expand(N,M).double()
    ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()

    alternate_neg = (-ones_)**(range_)
    ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
    ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
    as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
    bs_o = binomial_coefficient * as_o
    r = alternate_neg * bs_o
    # print("bs_o.size ", ins.size())
    # print("sigma.size", sigma.size())
    # print("erf ", torch.erf(ins))
    logv = torch.log(ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1))))
    # print("zeta log ",logv )
    logv.sum().backward()
    log_grad = sigma.grad
    # print("logv.grad ", sigma.grad)
    # print("ins.grad ", ins.grad)
    # print("log_grad ",log_grad)
    return sigma.grad.data

def gaussianPDF(x, mu, sigma, distance=pf.distance, norm_func=zeta):
    # norm_func = zeta
    # print(x.shape, mu.shape)
    N, D, M = x.shape + (mu.shape[0],)
    # print("N, M, D ->", N, M, D)
    # x <- N x M x D
    # mu <- N x M x D
    # sigma <- N x M
    x_rd = x.unsqueeze(1).expand(N, M, D)
    mu_rd = mu.unsqueeze(0).expand(N, M, D)
    sigma_rd = sigma.unsqueeze(0).expand(N, M)
    # computing numerator
    num = torch.exp(-((distance(x_rd, mu_rd)**2))/(2*(sigma_rd)**2))
    # print("num mean ",num.mean())
    den = norm_func(sigma)
    # print("den mean ",den.mean() )
    # print("sigma",num)
    # print("den ", den)
    # print("pdf max ", (num/den.unsqueeze(0).expand(N, M)).max())
    return num/den.unsqueeze(0).expand(N, M)