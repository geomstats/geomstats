"""Principal Component Analysis on Manifolds.

Lead author: Nina Miolane.
"""

import numbers
from math import log

from scipy.special import gammaln
from sklearn.decomposition._base import _BasePCA
from sklearn.utils.extmath import stable_cumsum, svd_flip

import geomstats.backend as gs
from geomstats.geometry._hyperbolic import HyperbolicMetric
from geomstats.geometry.hyperbolic import Hyperbolic
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
        raise ValueError("The tested rank cannot exceed the rank of the dataset")

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
        if self.space.point_ndim == 2:
            if (
                gs.all(Matrices.is_square(tangent_vecs))
                and Matrices.is_symmetric(tangent_vecs).all()
            ):
                X = SymmetricMatrices.basis_representation(tangent_vecs)
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
            if gs.all(Matrices.is_square(self.base_point_)) and gs.all(
                Matrices.is_symmetric(self.base_point_)
            ):
                scores = SymmetricMatrices(
                    self.base_point_.shape[-1]
                ).matrix_representation(scores)
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
            if gs.all(Matrices.is_square(tangent_vecs)) and gs.all(
                Matrices.is_symmetric(tangent_vecs)
            ):
                X = SymmetricMatrices.basis_representation(tangent_vecs)
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


class HyperbolicPlanePCA(_BasePCA):
    """Exact Principal Geodesic Analysis in the hyperbolic plane.

    The first principal component is computed by finding the direction
    in a unit ball around the mean that minimizes the variance of the
    projections on the induced geodesic. The projections are given by
    closed form expressions in extrinsic coordinates. The second principal
    component is the direction at the mean that is orthogonal to the first
    principal component.

    References:
    ----------
    .. [CSV2016] R. Chakraborty, D. Seo, and B. C. Vemuri,
        "An efficient exact-pga algorithm for constant curvature manifolds."
        Proceedings of the IEEE conference on computer vision and pattern
        recognition. 2016.
    """

    def __init__(self, space, n_vec=100):
        self.space = space
        self.half_space = Hyperbolic(2, coords_type="half-space")
        self.space_ext = Hyperbolic(2, coords_type="extrinsic")
        self.n_vec = n_vec

    def _project_on_geodesic_extrinsic(self, point_ext, mean_ext, vector_ext):
        """Project on geodesic in extrinsic coordinates.

        Project point onto geodesic going through mean in direction of vector.
        """
        inner_prod_1 = self.space_ext.metric.inner_product(point_ext, vector_ext)
        inner_prod_2 = self.space_ext.metric.inner_product(point_ext, mean_ext)
        norm_v = self.space_ext.metric.norm(vector_ext, mean_ext)
        dist_to_proj = gs.arctanh(-inner_prod_1 / inner_prod_2 / norm_v)
        proj = gs.einsum("...,i->...i", gs.cosh(dist_to_proj), mean_ext) + gs.einsum(
            "...,i->...i", gs.sinh(dist_to_proj), vector_ext / norm_v
        )
        return proj

    def _fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Training data in the hyperbolic plane. If the space is
            the Poincare half-space or Poincare ball, n_features is
            2. If it is the hyperboloid, n_features is 3.
        y : Ignored (Compliance with scikit-learn interface)

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        estimator = FrechetMean(space=self.space).fit(X)
        self.mean_ = estimator.estimate_

        mean_half_space = self.space.to_coordinates(self.mean_, "half-space")
        mean_ext = self.space.to_coordinates(self.mean_, "extrinsic")
        X_ext = self.space.to_coordinates(X, "extrinsic")

        angles_half_space = gs.linspace(0.0, 2 * gs.pi, self.n_vec)
        angles_half_space = gs.expand_dims(angles_half_space, axis=1)
        vectors_half_space = gs.hstack(
            (gs.cos(angles_half_space), gs.sin(angles_half_space))
        )
        norms = self.half_space.metric.norm(vectors_half_space, mean_half_space)
        vectors_half_space = gs.einsum("ij,i->ij", vectors_half_space, 1 / norms)
        vectors_ext = self.space.half_space_to_extrinsic_tangent(
            vectors_half_space, mean_half_space
        )

        def variance_of_projections(pt_ext, mn_ext, vec_ext):
            projections = self._project_on_geodesic(pt_ext, mn_ext, vec_ext)
            costs = self.space_ext.metric.dist(mn_ext, projections) ** 2
            return gs.sum(costs)

        costs = [
            variance_of_projections(X_ext, mean_ext, vec_ext) for vec_ext in vectors_ext
        ]
        axis_1 = vectors_half_space[gs.argmin(costs)]
        axis_2 = gs.array([-axis_1[1], axis_1[0]])
        components_half_space = gs.stack((axis_1, axis_2))
        self.components_ = self.space.to_tangent_coordinates(
            components_half_space, "half-space"
        )
        return self

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Training data in the hyperbolic plane. If the space is
            the Poincare half-space or Poincare ball, n_features is
            2. If it is the hyperboloid, n_features is 3.
        y : Ignored (Compliance with scikit-learn interface)

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Project X on the principal components.

        Parameters
        ----------
         X : array-like, shape=[..., n_features]
             Training data in the hyperbolic plane. If the space is
             the Poincare half-space or Poincare ball, n_features is
             2. If it is the hyperboloid, n_features is 3.
         y : Ignored (Compliance with scikit-learn interface)

        Returns
        -------
        X_1 : array-like, shape=[..., n_features]
            Projections of the data on the first component.
        X_2 : array-like, shape=[..., n_features]
            Projections of the data on the second component.
        """
        self._fit(X)
        axis_1, axis_2 = self.components_
        axis_1_ext = self.space.to_tangent_coordinates(axis_1, self.mean_, "extrinsic")
        axis_2_ext = self.space.to_tangent_coordinates(axis_2, self.mean_, "extrinsic")
        X_ext = self.space.to_coordinates(X, "extrinsic")
        mean_ext = self.space.to_coordinates(self.mean_, "extrinsic")

        proj1_ext = self._project_on_geodesic(X_ext, mean_ext, axis_1_ext)
        proj2_ext = self._project_on_geodesic(X_ext, mean_ext, axis_2_ext)
        X_1 = self.space.from_coordinates(proj1_ext, "extrinsic")
        X_2 = self.space.from_coordinates(proj2_ext, "extrinsic")

        return X_1, X_2


class ExactPGA(_BasePCA):
    r"""Exact Principal Geodesic Analysis.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    """

    def __new__(cls, space, **kwargs):
        """Interface for instantiating proper algorithm."""
        if isinstance(space.metric, HyperbolicMetric) and space.dim == 2:
            return HyperbolicPlanePCA(space, **kwargs)

        else:
            raise NotImplementedError(
                "Exact PGA is only implemented for the two-dimensional "
                "hyperbolic space."
            )
