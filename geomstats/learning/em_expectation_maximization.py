"""Expectation maximisation algorithm."""

from random import randint

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
import torch
from torch import nn
from geomstats.learning.frechet_mean import FrechetMean
import math
import os

EM_CONV_RATE = 1e-4
DEFAULT_MAX_ITER = 100
DEFAULT_LR = 5e-2
DEFAULT_TAU = 1e-4
ZETA_CST = gs.sqrt(gs.pi/2)


class RiemannianEM(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self,
                 riemannian_metric,
                 n_gaussian=8,
                 init='random',
                 tol=1e-2,
                 mean_method='default',
                 point_type='vector',
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
        self.point_type = point_type

    def update_weights(self, wik, g_index=-1):
        """ Weights update function
        """



        if (g_index > 0):
            self.mixture_coefficients[g_index] = gs.mean(wik[:, g_index])
        else:
            self.mixture_coefficients = gs.mean(wik,0)



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

        #TODO Adapt to big number of gaussians
        # if too much gaussian we compute mean for each gaussian separately (To avoid too large memory)
        # if(M>40):
        #     for g_index in range(M//40 + (1 if(M%40 != 0) else 0)):
        #         from_ = g_index * 40
        #         to_ = min((g_index+1) * 40, M)
        #         # print(from_, to_, " to_from :")
        #
        #         zz = z.unsqueeze(1).expand(N, to_-from_, D)
        #         self.means[from_:to_] = barycenter(zz, wik[:, from_:to_], lr_mu, tau_mu, max_iter=max_iter,
        #                                            verbose=True, normed=True).squeeze()
        # else:

        # if(g_index>0):
        #     self.means[g_index] = barycenter(data, wik[:, g_index], lr_mu, tau_mu, max_iter=max_iter, normed=True).squeeze()
        # else:
        #     self.means = barycenter(data.unsqueeze(1).expand(N, M, D), wik, lr_mu, tau_mu, max_iter=max_iter, normed=True).squeeze()


    def update_variances(self, z, wik, g_index=-1):
        """Variances update function"""
        with torch.no_grad():
            N, D, M = z.shape + (self.means.shape[0],)
            if (g_index > 0):
                dtm = ((distance(z, self.means[:, g_index].expand(N)) ** 2) * wik[:, g_index]).sum() / wik[:,
                                                                                                           g_index].sum()
                self.variances[:, g_index] = (self.normalization_factor.phi(dtm)).data.numpy()
            else:
                dtm = ((distance(z.unsqueeze(1).expand(N, M, D),
                                 self.means.unsqueeze(0).expand(N, M, D)) ** 2) * wik).sum(0) / wik.sum(0)
                # print("dtms ", dtm.size())
                self.variances = (self.normalization_factor.phi(dtm)).data.numpy()

    def _expectation(self, data):
        """Compute weights_ik given the data, means and variances"""

        probability_distribution_function = gaussianPDF(data,
                                                        self.means,
                                                        self.variances,
                                                        distance= distance,
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

        self.update_weights(wik)

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
        self.means = torch.from_numpy(self.means)
        self.update_variances(torch.from_numpy(data), torch.from_numpy(wik))
        self.means = self.means.data.numpy()
        if(self.verbose):
            print("sigma ", self.variances)
        if(self.variances.mean() != self.variances.mean()):
            print("UPDATE : sigma contain not a number elements")
            quit()

    def fit(self,
            data,
            max_iter= DEFAULT_MAX_ITER,
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
        if(self.init =='random'):

            self._dimension = data.shape[-1]

            self.means = (gs.random.rand(self.n_gaussian, self._dimension) - 0.5) / self._dimension
            self.variances = gs.random.rand(self.n_gaussian) / 10 + 0.8
            self.mixture_coefficients = gs.ones(self.n_gaussian) / self.n_gaussian
            posterior_probabilities = gs.ones((data.shape[0], self.means.shape[0]))

            #TODO Write properly ZetaPhiStorage
            self.normalization_factor = ZetaPhiStorage(gs.arange(5e-2, 2., 0.001), self._dimension)

        if (self.verbose):
            print("Number of data samples", data.shape[0])
            print("Number of Gaussian distribution", self._dimension)
            print("Initial Variances", self.variances)
            print("Initial Mixture Weights", self.mixture_coefficients)
            print("Initial Weights", posterior_probabilities)

        for epoch in range(max_iter):
            old_wik = posterior_probabilities

            posterior_probabilities = self._expectation(data)
            condition = gs.mean(gs.abs(old_wik - posterior_probabilities))
            if(condition < 1e-4 and epoch>10):

                print('EM converged in ', epoch, 'iterations')
                return self.means, self.variances, self.mixture_coefficients

            self._maximization(data, posterior_probabilities, lr_mu=lr_mu, tau_mu=tau_mu)

        print('WARNING: EM did not converge')

        return self.means, self.variances, self.mixture_coefficients

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
        #TODO Thomas ou Hadi: Write prediction method to
        # label points with the cluster maximising the likelihood
        belongs = None
        return belongs


class ZetaPhiStorage(object):
    """A class for computing the normalization factor."""

    def __init__(self, sigma, dimension):

        self.dimension = dimension
        sigma = torch.from_numpy(sigma)
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


        #sigma_cube = self.sigma ** 3

        #factor_normalization, log_grad_zeta = zeta_dlogzetat(self.sigma.data.numpy(), dimension)

        #self.phi_inv_var = sigma_cube * log_grad_zeta

        #self.phi_inv_var = (self.sigma ** 3 * log_grad_zeta(self.sigma, dimension)).detach()
        self.phi_inv_var = (self.sigma ** 3 * log_grad_zeta(self.sigma, dimension)).detach()

        print(self.phi_inv_var.type())

    def zeta(self, sigma):
        N, P = sigma.shape[0], self.sigma.shape[0]
        ref = self.sigma.unsqueeze(0).expand(N, P)
        val = sigma.unsqueeze(1).expand(N, P)
        values, index = torch.abs(ref - val).min(-1)

        return self.m_zeta_var[index]

    def zeta_numpy(self, sigma):
        N, P = sigma.shape[0], self.sigma.shape[0]

        sigma_self = self.sigma.data.numpy()
        ref = gs.expand_dims(sigma_self,0)
        ref = gs.repeat(ref, N, axis= 0)
        val = gs.expand_dims(sigma, 1)
        val = gs.repeat(val,P, axis = 1)

        difference = gs.abs(ref-val)

        index = gs.argmin(difference,axis = -1)

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

CST_FOR_ERF = 8.0 / (3.0 * gs.pi) * (gs.pi - 3.0) / (4.0 - gs.pi)
def erf_approx(x):
    return gs.sign(x)*gs.sqrt(1 - gs.exp(-x * x * (4 / gs.pi + CST_FOR_ERF * x * x) / (1 + CST_FOR_ERF * x * x)))

def new_zeta(x, N):
    x = x.data.numpy()

    sigma = x

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
    as_o_gs = (1+erf_approx(ins_gs)) * gs.exp(ins_squared_gs)
    bs_o_gs = binomial_coefficient * as_o_gs
    r_gs = alternate_neg * bs_o_gs

    zeta = ZETA_CST * sigma * r_gs.sum(0) * (1/(2**(N-1)))

    return zeta


def log_grad_zeta(x, N):
    sigma = nn.Parameter(x)
    # print(sigma.grad)
    binomial_coefficient=None
    M = sigma.shape[0]
    sigma_u = sigma.unsqueeze(0).expand(N,M)
    if(binomial_coefficient is None):
        # we compute coeficient
        v = torch.arange(N)
        v[0] = 1
        n_fact = v.prod()
        k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
        nmk_fact = k_fact.flip(0)
        binomial_coefficient = n_fact/(k_fact * nmk_fact)
    binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).double()
    range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()
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
    #return ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1))),sigma.grad.data
    #return ZETA_CST * sigma * r.sum(0) * (1 / (2 ** (N - 1))) , log_grad
    return sigma.grad.data

def gaussianPDF(data, means, variances, distance, norm_func, metric):
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


def distance(x, y):
    return PoincareDistance2.apply(x, y)

class PoincareDistance2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        with torch.no_grad():
            x_norm = torch.clamp(torch.sum(x ** 2, dim=-1), 0, 1-5e-4)
            y_norm = torch.clamp(torch.sum(y ** 2, dim=-1), 0, 1-5e-4)
            d_norm = torch.sum((x-y) ** 2, dim=-1)
            cc = 1+2*d_norm/((1-x_norm)*(1-y_norm))
            dist = torch.log(cc + torch.sqrt(cc**2-1))
            ctx.save_for_backward( x, y, dist)
            return  dist
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            x, y, dist = ctx.saved_tensors
            dist_unsqueeze = dist.unsqueeze(-1).expand_as(x)
            # print(y)
            # print(dist)
            # print(log(x, y))
            # print(grad_output)
            res_x, res_y =  (- (log(x, y)/(dist_unsqueeze)) * grad_output.unsqueeze(-1).expand_as(x),
                     - (log(y, x)/(dist_unsqueeze)) * grad_output.unsqueeze(-1).expand_as(x))
            # print(res_y)
            if((dist == 0).sum() != 0):
                # it exist example having same representation
                res_x[dist == 0 ] = 0
                res_y[dist == 0 ] = 0
            return res_x, res_y

def add(x, y):
    nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x) *1
    ny = torch.sum(y ** 2, dim=-1, keepdim=True).expand_as(x) *1
    xy = (x * y).sum(-1, keepdim=True).expand_as(x)*1
    return ((1 + 2*xy+ ny)*x + (1-nx)*y)/(1+2*xy+nx*ny)

def log(k, x):
    kpx = add(-k,x)
    # print("START KPX")
    # print((-k).sum(0))
    # print((x).sum(0))
    # print(kpx.sum(0))
    # print("END KPX")
    norm_kpx = kpx.norm(2,-1, keepdim=True).expand_as(kpx)
    norm_k = k.norm(2,-1, keepdim=True).expand_as(kpx)
    res = (1-norm_k**2)* ((atanh(norm_kpx))) * (kpx/norm_kpx)
    if(0 != len((norm_kpx==0).nonzero())):
        res[norm_kpx == 0] = 0
    return res



def exp(k, x):
    norm_k = k.norm(2,-1, keepdim=True).expand_as(k)*1
    lambda_k = 1/(1-norm_k**2)
    norm_x = x.norm(2,-1, keepdim=True).expand_as(x)*1
    direction = x/norm_x
    factor = torch.tanh(lambda_k * norm_x)
    res = add(k, direction*factor)
    if(0 != len((norm_x==0).nonzero())):
        res[norm_x == 0] = k[norm_x == 0]
    return res

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

##### 
SQRT_2 = math.sqrt(2.)


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


import numpy as np

if __name__ == "__main__":
    a, b = zeta_dlogzetat(gs.array([[0.5, 1.2 , 1.5]]), 4)
    print(a,b)
    a_o, b_o = log_grad_zeta(torch.Tensor([0.5, 1.2, 1.5]), 4)
    print(a_o, b_o)