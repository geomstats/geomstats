"""Expectation maximisation algorithm."""

from random import randint

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
import torch
from torch import nn
import math




EM_CONV_RATE = 1e-4
DEFAULT_MAX_ITER = 100
DEFAULT_LR = 5e-2
DEFAULT_TAU = 1e-4
ZETA_CST = gs.sqrt(gs.pi/2)


class RawDataloader(object):
    def __init__(self, dataset, batch_size=200, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        if(self.shuffle):
            self.indexed = torch.randperm(len(self.dataset))
        self.indexed = torch.arange(len(self.dataset))
        self.ci = 0
        return self

    def __len__(self):
        return len(self.dataset)//self.batch_size +  (0 if(len(self.dataset)%self.batch_size == 0) else 1)

    def _next_index(self):
        if(self.ci >= (len(self))):
            raise StopIteration
        value = self.dataset[self.indexed[self.ci*self.batch_size: min((self.ci+1)*self.batch_size, len(self.dataset)) ]]
        self.ci += 1
        return value

    def __next__(self):
        return self._next_index()

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

    def update_weights(self,X, wik, g_index=-1):
        """ Weights update function
        """
        with torch.no_grad():
            # get omega mu

            if(g_index > 0):
                self.mixture_coefficients[g_index] = wik[:, g_index].mean()
            else:
                self.mixture_coefficients = wik.mean(0)

    def update_means(self, z, wik, lr_mu, tau_mu, g_index=-1, max_iter=150):
        """ Means update functions"""
        N, D, M = z.shape + (wik.shape[-1],)
        # if too much gaussian we compute mean for each gaussian separately (To avoid too large memory)
        if(M>40):
            for g_index in range(M//40 + (1 if(M%40 != 0) else 0)):
                from_ = g_index * 40
                to_ = min((g_index+1) * 40, M)
                # print(from_, to_, " to_from :")

                zz = z.unsqueeze(1).expand(N, to_-from_, D)
                self.means[from_:to_] = barycenter(zz, wik[:, from_:to_], lr_mu, tau_mu, max_iter=max_iter,
                                                   verbose=True, normed=True).squeeze()
        else:
            if(g_index>0):
                self.means[g_index] = barycenter(z, wik[:, g_index], lr_mu, tau_mu, max_iter=max_iter, normed=True).squeeze()
            else:
                self.means = barycenter(z.unsqueeze(1).expand(N, M, D), wik, lr_mu, tau_mu, max_iter=max_iter, normed=True).squeeze()
        # print("1",self._mu)
    def update_variances(self, z, wik, g_index=-1):
        """Variances update function"""
        with torch.no_grad():
            N, D, M = z.shape + (self.means.shape[0],)
            if (g_index > 0):
                dtm = ((distance(z, self.means[:, g_index].expand(N)) ** 2) * wik[:, g_index]).sum() / wik[:,
                                                                                                           g_index].sum()
                self.variances[:, g_index] = self.normalization_factor.phi(dtm)
            else:
                dtm = ((distance(z.unsqueeze(1).expand(N, M, D),
                                 self.means.unsqueeze(0).expand(N, M, D)) ** 2) * wik).sum(0) / wik.sum(0)
                # print("dtms ", dtm.size())
                self.variances = self.normalization_factor.phi(dtm)

    def _expectation(self, data):
        """Compute weights_ik given the data, means and variances"""

        probability_distribution_function = gaussianPDF(data,
                                                        self.means,
                                                        self.variances,
                                                        distance= distance,
                                                        norm_func=self.normalization_factor.zeta)

        if (probability_distribution_function.mean() !=
                probability_distribution_function.mean()):
            print("EXPECTATION : pdf contain not a number elements")
            quit()

        p_pdf = probability_distribution_function * \
                self.mixture_coefficients.unsqueeze(0).expand_as(probability_distribution_function)
        wik = gs.ones((data.shape[0], self.means.shape[0]))
        if (p_pdf.sum(-1).min() <= 1e-15):
            if (self._verbose):
                print("EXPECTATION : pdf.sum(-1) contain zero for ", (p_pdf.sum(-1) <= 1e-15).sum().item(), "items")
            p_pdf[p_pdf.sum(-1) <= 1e-15] = 1

        wik = p_pdf / p_pdf.sum(-1, keepdim=True).expand_as(probability_distribution_function)
        if (wik.mean() != wik.mean()):
            print("EXPECTATION : wik contain not a number elements")
            quit()
        # print(wik.mean(0))
        if (wik.sum(1).mean() <= 1 - 1e-4 and wik.sum(1).mean() >= 1 + 1e-4):
            print("EXPECTATION : wik don't sum to 1")
            print(wik.sum(1))
            quit()
        return wik

    def _maximization(self, data, wik, lr_mu, tau_mu, max_iter = math.inf ):
        """Given the weights and data, will update the means and variances."""
        self.update_weights(data, wik)
        if(self.mixture_coefficients.mean() != self.mixture_coefficients.mean()):
            print("UPDATE : w contain not a number elements")
            quit()
        # print("w", self._w)
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

            self._dimension = data.size(-1)
            self.means = (gs.random.rand(self.n_gaussian, self._dimension) - 0.5) / self._dimension
            self.variances = gs.random.rand(self.n_gaussian) / 10 + 0.8
            self.mixture_coefficients = gs.ones(self.n_gaussian) / self.n_gaussian
            posterior_probabilities = gs.ones((data.size(0), self.means.size(0)))

            #TODO Write properly ZetaPhiStorage
            self.normalization_factor = ZetaPhiStorage(torch.arange(5e-2, 2., 0.001), self._dimension)


        if (self.verbose):
            print("Number of data samples", data.shape[0])
            print("Number of Gaussian distribution", self._dimension)
            print("Initial Variances", self.variances)
            print("Initial Mixture Weights", self.mixture_coefficients)
            print("Initial Weights", posterior_probabilities)

        for epoch in range(max_iter):
            old_wik = posterior_probabilities

            posterior_probabilities = self._expectation(data)
            if((old_wik - posterior_probabilities).abs().mean() < 1e-4 and epoch>10):

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

    sigma = nn.Parameter(x)
    # binomial_coefficient=None
    # M = sigma.shape[0]
    #
    # sigma_u = gs.expand_dims(sigma,0)

    binomial_coefficient = None
    M = sigma.shape[0]
    sigma_u = sigma.unsqueeze(0).expand(N, M)

    if(binomial_coefficient is None):
         # we compute coeficient
         v = torch.arange(N)
         v[0] = 1
         n_fact = v.prod()

         k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
         nmk_fact = k_fact.flip(0)
         # print(nmk_fact)
         binomial_coefficient = n_fact / (k_fact * nmk_fact)
    binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N, M).double()
    #N = torch.from_numpy(N)
    range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()
    ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()

    alternate_neg = (-ones_)**(range_)

    ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
    ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
    as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
    bs_o = binomial_coefficient * as_o
    r = alternate_neg * bs_o


    zeta = ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1)))

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
    return sigma.grad.data

def gaussianPDF(x, mu, sigma, distance, norm_func):
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

def barycenter(z, wik=None, lr=5e-2, tau=5e-3, max_iter=math.inf, distance=distance, normed=False,
               init_method="default", verbose=False):
    with torch.no_grad():
        if (wik is None):
            wik = 1.
            # barycenter = z.mean(0, keepdim=True)
            barycenter = z.mean(0, keepdim=True) * 0
        else:

            wik = wik.unsqueeze(-1).expand_as(z)
            if (init_method == "global_mean"):
                print("Bad init selected")
                barycenter = z.mean(0, keepdim=True)
            else:
                barycenter = (z * wik).sum(0, keepdim=True) / wik.sum(0)

        if (len(z) == 1):
            return z
        iteration = 0
        cvg = math.inf
        # print("barycenter_init", barycenter)
        while (cvg > tau and max_iter > iteration):

            iteration += 1
            if (type(wik) != float):
                grad_tangent = 2 * log(barycenter.expand_as(z), z) * wik
                nan_values = (~(barycenter == barycenter))
                # print("\n\n A At least one barycenter is Nan : ")
                # print(pf.log(barycenter.expand_as(z), z).sum(0))
                if (nan_values.squeeze().nonzero().size(0) > 0):
                    print("\n\n A At least one barycenter is Nan : ")
                    print(log(barycenter.expand_as(z), z).sum(0))
                    print("index of nan values ", nan_values.squeeze().nonzero())
                    quit()
                    # torch 1.3 minimum for this operation

                    # nan_values = (~(barycenter == barycenter))
                    # print("Condition value ", (barycenter == barycenter).float().mean().item())
                    print("index of nan values ", nan_values.squeeze().nonzero())
                    # print("wik of nan values ", wik[:, nan_values.squeeze()].mean(0))
                    # print("At iteration ", iteration)
            else:
                grad_tangent = 2 * log(barycenter.expand_as(z), z)

            # print(type(wik))
            if (normed):
                # print(grad_tangent.size())
                if (type(wik) != float):
                    # print(wik.sum(0, keepdim=True))
                    grad_tangent /= wik.sum(0, keepdim=True).expand_as(wik)
                else:
                    grad_tangent /= len(z)

            cc_barycenter = exp(barycenter, lr * grad_tangent.sum(0, keepdim=True))
            nan_values = (~(cc_barycenter == cc_barycenter))

            if (nan_values.squeeze().nonzero().size(0) > 0):
                print("\n\n C At least one barycenter is Nan exp update may contain 0: ")
                print(grad_tangent.sum(0, keepdim=True))
                quit()
                # torch 1.3 minimum for this operation

                # print("Condition value ", (cc_barycenter == cc_barycenter).float().mean().item())
                # print("index of nan values ", nan_values.squeeze().nonzero())
                # print("wik of nan values ", wik[:, nan_values.squeeze()].mean(0))
                # print("At iteration ", iteration)
            cvg = distance(cc_barycenter, barycenter).max().item()
            # print(cvg)
            barycenter = cc_barycenter
            if (cvg <= tau and verbose):
                print("Frechet Mean converged in ", iteration, " iterations")
        if (type(wik) != float):
            # # to debug ponderate version
            # print(cvg, iteration, max_iter)
            pass
        # print("BARYCENTERS -> ", barycenter)
        return barycenter

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