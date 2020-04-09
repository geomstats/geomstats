"""Expectation maximisation algorithm."""

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.gaussian_distribution import GaussianDistribution,Normalization_Factor_Storage


EM_CONV_RATE = 1e-4
MINIMUM_EPOCHS = 10
DEFAULT_MAX_ITER = 100
DEFAULT_LR = 5e-2
DEFAULT_CONV_FACTOR = 1e-4
DEFAULT_TOL = 1e-2
ZETA_LOWER_BOUND = 5e-2
ZETA_UPPER_BOUND = 2.
ZETA_STEP = 0.001
PDF_TOL = 1e-15
SUM_CHECK_PDF = 1e-4

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
        """Updates the posterior probabilities"""

        probability_distribution_function = GaussianDistribution.gaussian_pdf(data,
                                                                              self.means,
                                                                              self.variances,
                                                                              norm_func=self.normalization_factor.normalisation_factor,
                                                                              metric =self.riemannian_metric)


        if (probability_distribution_function.mean() !=
                probability_distribution_function.mean()):
            print('EXPECTATION : Probability distribution function'
                  'contain elements that are not numbers')
            quit()

        prob_distrib_expand = gs.repeat(gs.expand_dims(self.mixture_coefficients,0),
                                        len(probability_distribution_function),
                                        axis = 0)

        num_normalized_pdf = probability_distribution_function* \
                   prob_distrib_expand

        valid_pdf_condition = gs.amin(gs.sum(num_normalized_pdf, -1))

        if (valid_pdf_condition <= PDF_TOL):

            if (self._verbose):
                print("EXPECTATION : Probability distribution function "
                      "contain zero for ",
                      gs.sum(gs.sum(num_normalized_pdf,-1) <= PDF_TOL))

            num_normalized_pdf[gs.sum(num_normalized_pdf,-1) <= PDF_TOL] = 1

        denum_normalized_pdf = gs.repeat(gs.sum(num_normalized_pdf,-1, keepdims=True),
                                         num_normalized_pdf.shape[-1],
                                         axis = 1)

        posterior_probabilities = num_normalized_pdf / denum_normalized_pdf

        if (gs.mean(posterior_probabilities) != gs.mean(posterior_probabilities)):

            print('EXPECTATION : posterior probabilities'
                  'contain elements that are not numbers')
            quit()


        if gs.mean(gs.sum(posterior_probabilities,1)) <= 1-SUM_CHECK_PDF \
                and gs.mean(gs.sum(posterior_probabilities,1)) >= 1+SUM_CHECK_PDF:

            print('EXPECTATION : posterior probabilities'
                  'don\'t sum to 1')
            print(gs.sum(posterior_probabilities,1))
            quit()

        return posterior_probabilities

    def _maximization(self,
                      data,
                      posterior_probabilities,
                      lr_means,
                      conv_factor_mean,
                      max_iter = gs.inf):
        """Update function for the means and variances."""

        self.update_posterior_probabilities(posterior_probabilities)

        if(gs.mean(self.mixture_coefficients)
                != gs.mean(self.mixture_coefficients)):
            print('UPDATE : mixture coefficients '
                  'contain elements that are not numbers')
            quit()

        self.update_means(data,
                          posterior_probabilities,
                          lr_mu=lr_means,
                          tau_mu=conv_factor_mean,
                          max_iter=max_iter)

        if(self.means.mean() != self.means.mean()):
            print('UPDATE : means contain'
                  'not a number elements')
            quit()

        self.update_variances(data, posterior_probabilities)

        if(self.variances.mean() != self.variances.mean()):
            print('UPDATE : variances contain'
                  'not a number elements')
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
            self.normalization_factor = Normalization_Factor_Storage(gs.arange(ZETA_LOWER_BOUND,
                                                                               ZETA_UPPER_BOUND,
                                                                               ZETA_STEP),
                                                                     self._dimension)

        else:
            print('Initialisation method not yet implemented')
            quit()

        if (self.verbose):
            print('Number of data samples', data.shape[0])
            print('Dimensions', self._dimension)

        for epoch in range(max_iter):
            old_posterior_probabilities = posterior_probabilities

            posterior_probabilities = self._expectation(data)

            condition = gs.mean(gs.abs(old_posterior_probabilities
                                       - posterior_probabilities))
            if(condition < EM_CONV_RATE and epoch > MINIMUM_EPOCHS):

                print('EM converged in ', epoch, 'iterations')
                return self.means, self.variances, self.mixture_coefficients

            self._maximization(data,
                               posterior_probabilities,
                               lr_means=lr_mean,
                               conv_factor_mean=conv_factor_mean)

        print('WARNING: EM did not converge'
              'Please increase MINIMUM_EPOCHS.')

        return self.means, self.variances, self.mixture_coefficients

    def predict(self, data):

        """Predict for each data point its Gaussian component.

        Given each Gaussin of the computed mixture model,
        this function computes for each data point
        the probability to belong to the Gaussian then
        takes the maximum probability taking into account the weight
         of the Gaussian to label the data point.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
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
