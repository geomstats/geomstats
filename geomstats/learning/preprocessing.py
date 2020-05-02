"""Transformer for manifold-valued data."""

from sklearn.base import BaseEstimator, TransformerMixin

import geomstats.backend as gs
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.frechet_mean import FrechetMean

EPSILON = 1e-4


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
    geometry : {Manifold ,LieGroup or RiemannianMetric}
        Metric or Lie group to use to compute the log and exp. If a Lie group
        is passed, its group exp/log will be used, which don't necessarily
        correspond to a Riemannian metric. To use a `metric` on the Lie group,
        explicitly pass `geometry=metric`
    **kwargs : key-word arguments for the FrechetMean/ExponentialBarycenter
        estimator.
    """

    def __init__(self, geometry, **kwargs):

        if isinstance(geometry, LieGroup):
            self._used_geometry = geometry
            self.estimator = ExponentialBarycenter(
                group=self._used_geometry, **kwargs)
        else:
            if hasattr(geometry, 'metric'):
                self._used_geometry = geometry.metric
            elif isinstance(geometry, RiemannianMetric):
                self._used_geometry = geometry
            else:
                raise ValueError('The input geometry must be either a '
                                 'Manifold equipped with a '
                                 'RiemannianMetric, or a RiemannianMetric or a'
                                 ' LieGroup')
            self.estimator = FrechetMean(metric=self._used_geometry, **kwargs)
        self.point_type = geometry.default_point_type
        self.geometry = geometry

    def fit(self, X, y=None, weights=None, base_point=None):
        """Compute the central point at which to take the log.

        This method is only used if `base_point=None` to compute the mean of
        the input data.

        Parameters
        ----------
        X : array-like, shape=[..., {dim, [n, n]}]
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Ignored
        weights : array-like, shape=[..., 1]
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
            self.estimator.fit(X, y, weights)
        return self

    def transform(self, X, base_point=None):
        """Lift data to a tangent space.

        Compute the logs of all data point and reshapes them to
        1d vectors if necessary. By default the logs are taken at the mean
        but any other base point can be passed. Any machine learning
        algorithm can then be used with the output array.

        Parameters
        ----------
        X : array-like, shape=[..., {dim, [n, n]}]
            Data to transform.
        y : Ignored (Compliance with scikit-learn interface)
        base_point : array-like, shape={dim, [n,n]}, optional (mean)
            Point on the manifold, the returned samples will be tangent
            vectors at the base point.

        Returns
        -------
        X_new : array-like, shape=[..., dim]
            Lifted data.
        """
        if base_point is None:
            base_point = self.estimator.estimate_

            if self.estimator.estimate_ is None:
                raise RuntimeError('fit needs to be called first or a '
                                   'base_point passed.')

        tangent_vecs = self._used_geometry.log(X, base_point=base_point)

        if self.point_type == 'vector':
            return tangent_vecs

        if gs.all(Matrices.is_symmetric(tangent_vecs)):
            X = SymmetricMatrices.to_vector(
                tangent_vecs)
        elif gs.all(Matrices.is_skew_symmetric(tangent_vecs)):
            X = SkewSymmetricMatrices(
                tangent_vecs.shape[-1]).basis_representation(tangent_vecs)
        else:
            X = gs.reshape(tangent_vecs, (len(X), - 1))

        return X

    def inverse_transform(self, X, base_point=None):
        """Reconstruction of X.

        The reconstruction will match X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape=[..., dim]
            New data, where dim is the dimension of the manifold data belong
            to.
        base_point : array-like, shape={dim, [n,n]}, optional (mean)
            Point on the manifold, where the input samples are tangent
            vectors.

        Returns
        -------
        X_original : array-like, shape=[..., {dim, [n, n]}
            Data lying on the manifold.
        """
        if base_point is None:
            base_point = self.estimator.estimate_

            if self.estimator.estimate_ is None:
                raise RuntimeError('fit needs to be called first or a '
                                   'base_point passed.')

        if self.point_type == 'matrix':
            n_base_point = base_point.shape[-1]
            n_vecs = X.shape[-1]
            dim_sym = int(n_base_point * (n_base_point + 1) / 2)
            dim_skew = int(n_base_point * (n_base_point - 1) / 2)

            if gs.all(Matrices.is_symmetric(base_point)) and dim_sym == n_vecs:
                tangent_vecs = SymmetricMatrices(
                    base_point.shape[-1]).from_vector(X)
            elif dim_skew == n_vecs:
                tangent_vecs = SkewSymmetricMatrices(
                    dim_skew).matrix_representation(X)
            else:
                dim = base_point.shape[-1]
                tangent_vecs = gs.reshape(X, (len(X), dim, dim))
        else:
            tangent_vecs = X
        return self._used_geometry.exp(tangent_vecs, base_point)
