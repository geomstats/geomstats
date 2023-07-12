"""Principal Component Analysis on Manifolds.

Lead author: Nina Miolane.
"""

import numbers
from math import log

from scipy.special import gammaln
from sklearn.decomposition._base import _BasePCA
from sklearn.utils.extmath import stable_cumsum, svd_flip

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.frechet_mean import FrechetMean


def _assess_dimension_(spectrum, rank, n_samples, n_features):
    """Compute the likelihood of a rank ``rank`` dataset.

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.

    Parameters
    ----------
    spectrum : array of shape (n)
        Data spectrum.
    rank : int
        Tested rank value.
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.

    Returns
    -------
    ll : float
        Log-likelihood.

    Notes
    -----
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    """
    if rank > len(spectrum):
        raise ValueError("The tested rank cannot exceed the rank of the" " dataset")

    pu = -rank * log(2.0)
    for i in range(rank):
        pu += gammaln((n_features - i) / 2.0) - log(gs.pi) * (n_features - i) / 2.0

    pl = gs.sum(gs.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.0

    if rank == n_features:
        pv = 0
        v = 1
    else:
        v = gs.sum(spectrum[rank:]) / (n_features - rank)
        pv = -gs.log(v) * n_samples * (n_features - rank) / 2.0

    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = log(2.0 * gs.pi) * (m + rank + 1.0) / 2.0

    pa = 0.0
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log(
                (spectrum[i] - spectrum[j]) * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])
            ) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2.0 - rank * log(n_samples) / 2.0

    return ll


def _infer_dimension_(spectrum, n_samples, n_features):
    """Infer the dimension of a dataset of shape (n_samples, n_features).

    The dataset is described by its spectrum `spectrum`.
    """
    n_spectrum = len(spectrum)
    ll = gs.empty(n_spectrum)
    for rank in range(n_spectrum):
        ll[rank] = _assess_dimension_(spectrum, rank, n_samples, n_features)
    return ll.argmax()


class TangentPCA(_BasePCA):
    r"""Tangent Principal component analysis (tPCA).

    Linear dimensionality reduction using
    Singular Value Decomposition of the
    Riemannian Log of the data at the tangent space
    of the Frechet mean.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_components : int
        Number of principal components.
        Optional, default: None.

    Notes
    -----
    * Required geometry methods: `exp`, `log`.
    * If `base_point=None`, also requires `FrechetMean` required methods.
    * Lie groups can be used without a metric, but `base_point` or `mean_estimator`
     need to be specified.
    """

    def __init__(
        self,
        space,
        n_components=None,
        copy=True,
        whiten=False,
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        self.space = space
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        if hasattr(self.space, "metric"):
            self.mean_estimator = FrechetMean(space)
        else:
            self.mean_estimator = ExponentialBarycenter(space)

        self.base_point_ = None

    @property
    def _geometry(self):
        """Object where `exp` and `log` are defined."""
        if hasattr(self.space, "metric"):
            return self.space.metric

        return self.space

    def fit(self, X, y=None, base_point=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored (Compliance with scikit-learn interface)
        base_point : array-like, shape=[..., n_features], optional
            Point at which to perform the tangent PCA
            Optional, default to Frechet mean if None.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, base_point=base_point)
        return self

    def fit_transform(self, X, y=None, base_point=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored (Compliance with scikit-learn interface)
        base_point : array-like, shape=[..., n_features]
            Point at which to perform the tangent PCA
            Optional, default to Frechet mean if None.

        Returns
        -------
        X_new : array-like, shape=[..., n_components]
            Projected data.
        """
        U, S, _ = self._fit(X, base_point=base_point)

        U = U[:, : self.n_components_]

        U *= S[: self.n_components_]
        return U

    def transform(self, X, y=None):
        """Project X on the principal components.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored (Compliance with scikit-learn interface)

        Returns
        -------
        X_new : array-like, shape=[..., n_components]
            Projected data.
        """
        tangent_vecs = self._geometry.log(X, base_point=self.base_point_)
        if self.space.default_point_type == "matrix":
            if Matrices.is_symmetric(tangent_vecs).all():
                X = SymmetricMatrices.to_vector(tangent_vecs)
            else:
                X = gs.reshape(tangent_vecs, (len(X), -1))
        else:
            X = tangent_vecs
        X = X - self.mean_
        X_transformed = gs.matmul(X, gs.transpose(self.components_))
        return X_transformed

    def inverse_transform(self, X):
        """Low-dimensional reconstruction of X.

        The reconstruction will match X_original whose transform would be X
        if `n_components=min(n_samples, n_features)`.

        Parameters
        ----------
        X : array-like, shape=[..., n_components]
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original : array-like, shape=[..., n_features]
            Original data.
        """
        scores = self.mean_ + gs.matmul(X, self.components_)

        if self.space.point_ndim > 1:
            if gs.all(Matrices.is_symmetric(self.base_point_)):
                scores = SymmetricMatrices(self.base_point_.shape[-1]).from_vector(
                    scores
                )
            else:
                dim = self.base_point_.shape[-1]
                scores = gs.reshape(scores, (len(scores), dim, dim))
        return self._geometry.exp(scores, self.base_point_)

    def _fit(self, X, base_point=None):
        """Fit the model by computing full SVD on X.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored (Compliance with scikit-learn interface)
        base_point : array-like, shape=[..., n_features]
            Point at which to perform the tangent PCA.
            Optional, default to Frechet mean if None.

        Returns
        -------
        U, S, V : array-like
            Matrices of the SVD decomposition
        """
        if base_point is None:
            base_point = self.mean_estimator.fit(X).estimate_

        tangent_vecs = self._geometry.log(X, base_point=base_point)

        if self.space.point_ndim > 1:
            if gs.all(Matrices.is_symmetric(tangent_vecs)):
                X = SymmetricMatrices.to_vector(tangent_vecs)
            else:
                X = gs.reshape(tangent_vecs, (len(X), -1))
        else:
            X = tangent_vecs

        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components
        n_samples, n_features = X.shape

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                f"n_components={n_components} must be between 0 and "
                f"min(n_samples, n_features)={min(n_samples, n_features)} with "
                "svd_solver='full'"
            )
        elif n_components >= 1 and not isinstance(n_components, numbers.Integral):
            raise ValueError(
                f"n_components={n_components} must be of type int "
                "when greater than or equal to 1, "
                f"was of type={type(n_components)}"
            )

        # Center data - the mean should be 0 if base_point is the Frechet mean
        self.mean_ = gs.mean(X, axis=0)
        X -= self.mean_

        U, S, V = gs.linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        components_ = V

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = gs.copy(S)  # Store the singular values.

        # Postprocess the number of components required
        if n_components == "mle":
            n_components = _infer_dimension_(explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = gs.searchsorted(ratio_cumsum, n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.0

        self.base_point_ = base_point
        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = int(n_components)
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, S, V
