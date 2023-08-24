"""Transformer for manifold-valued data.

Lead author: Nicolas Guigui.
"""

from sklearn.base import BaseEstimator, TransformerMixin

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.frechet_mean import FrechetMean


class ToTangentSpace(BaseEstimator, TransformerMixin):
    """Lift data to a tangent space.

    Compute the logs of all data points and reshape them to
    1d vectors if necessary. This means that all the data points, that belong
    to a possibly non-linear manifold are lifted to one of the tangent space of
    the manifold, which is a vector space. By default, the mean of the data
    is computed (with the FrechetMean or the ExponentialBarycenter estimator,
    as appropriate) and the tangent space at the mean is used. Any other base
    point can be passed. The data points are then represented by the initial
    velocities of the geodesics that lead from base_point to each data point.
    Any machine learning algorithm can then be used with the output array.

    Parameters
    ----------
    space : Manifold
        Equipped manifold or unequipped space implementing `exp` and `log`.

    Notes
    -----
    * Required geometry methods: `log`, `exp`.
    """

    def __init__(self, space):
        self.space = space

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

    def fit(self, X, y=None, weights=None, base_point=None):
        """Compute the central point at which to take the log.

        This method is only used if `base_point=None` to compute the mean of
        the input data.

        Parameters
        ----------
        X : array-like, shape=[n_samples, {dim, [n, n]}]
            The training input samples.
        y : None
            Ignored.
        weights : array-like, shape=[n_samples, 1]
            Weights associated to the points.
            Optional, default: None
        base_point : array-like, shape=[{dim, [n, n]}]
            Point similar to the input data from which to compute the logs.
            Optional, default: None.

        Returns
        -------
        self : object
            Returns self.
        """
        if base_point is None:
            self.base_point_ = self.mean_estimator.fit(X, y, weights).estimate_

        else:
            self.base_point_ = base_point

        return self

    def transform(self, X):
        """Lift data to a tangent space.

        Compute the logs of all data point and reshapes them to
        1d vectors if necessary. By default the logs are taken at the mean
        but any other base point can be passed. Any machine learning
        algorithm can then be used with the output array.

        Parameters
        ----------
        X : array-like, shape=[n_samples, {dim, [n, n]}]
            Data to transform.

        Returns
        -------
        X_new : array-like, shape=[n_samples, dim]
            Lifted data.
        """
        if self.base_point_ is None:
            raise Exception("Not fitted")

        tangent_vecs = self._geometry.log(X, base_point=self.base_point_)

        if self.space.point_ndim == 1:
            return tangent_vecs

        if gs.all(Matrices.is_symmetric(tangent_vecs)):
            X = SymmetricMatrices.to_vector(tangent_vecs)

        elif gs.all(Matrices.is_skew_symmetric(tangent_vecs)):
            X = SkewSymmetricMatrices(tangent_vecs.shape[-1]).basis_representation(
                tangent_vecs
            )
        else:
            X = gs.reshape(tangent_vecs, (len(X), -1))

        return X

    def inverse_transform(self, X):
        """Reconstruction of X.

        The reconstruction will match X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim]
            New data, where dim is the dimension of the manifold data belong
            to.

        Returns
        -------
        X_original : array-like, shape=[n_samples, {dim, [n, n]}
            Data lying on the manifold.
        """
        if self.base_point_ is None:
            raise Exception("Not fitted")

        if self.space.point_ndim > 1:
            n_base_point = self.base_point_.shape[-1]
            n_vecs = X.shape[-1]
            dim_sym = int(n_base_point * (n_base_point + 1) / 2)
            dim_skew = int(n_base_point * (n_base_point - 1) / 2)

            if gs.all(Matrices.is_symmetric(self.base_point_)) and dim_sym == n_vecs:
                tangent_vecs = SymmetricMatrices(
                    self.base_point_.shape[-1]
                ).from_vector(X)
            elif dim_skew == n_vecs:
                tangent_vecs = SkewSymmetricMatrices(dim_skew).matrix_representation(X)
            else:
                dim = self.base_point_.shape[-1]
                tangent_vecs = gs.reshape(X, (len(X), dim, dim))
        else:
            tangent_vecs = X
        return self._geometry.exp(tangent_vecs, self.base_point_)
