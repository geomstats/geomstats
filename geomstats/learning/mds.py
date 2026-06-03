"""Multidimensional scaling (MDS)."""

import sklearn.manifold._mds as _mds
import sklearn.metrics.pairwise as mp
from sklearn.manifold import MDS as _MDS

import geomstats.backend as gs
from geomstats._sklearngs.manifold._classical_mds import ClassicalMDS
from geomstats._sklearngs.manifold._mds import smacof
from geomstats.geometry.manifold import Manifold

from ._sklearn import (
    SklearnInteropMixin,
    check_array_allow_nd,
    validate_data_skip_check_array,
)


def pairwise_dists(points, dist_fnc):
    """Compute the pairwise distance between points.

    Parameters
    ----------
    points : array-like, shape=[n_samples, dim]
        Set of points in the manifold.
    space : Manifold or PointSet


    Returns
    -------
    pairwise_dist_matrix : array-like, shape=[n_samples, n_samples]
        Pairwise distance matrix between all the points.
    """
    n_samples = len(points)

    pairwise_dist_matrix = gs.zeros((n_samples, n_samples))
    for i in range(n_samples - 1):
        dists = dist_fnc(points[i], points[i + 1 :])
        pairwise_dist_matrix[i, i + 1 :] = dists
        pairwise_dist_matrix[i + 1 :, i] = dists
    return pairwise_dist_matrix


class MDS(SklearnInteropMixin, _MDS):
    r"""Multidimensional scaling (MDS).

    MDS is used to translate pairwise distances of N objects into
    n points mapped into abstract Cartesian space, usually two-dimensional.

    In general, MDS doesn't need a metric, though this implementation will use one.

    Parameters
    ----------
    space : Manifold or PointSet
        Space equipped with a distance metric.
    n_components : int
        Number of dimensions in which to immerse the dissimilarities.

    Attributes
    ----------
    embedding_ : array-like, shape=[n_samples, n_components]
        Data transformed in the new space.

    Notes
    -----
    * For all other parameters see documentation in scikit-learn.
    * Required metric methods for general case:
        * `dist`

    References
    ----------
    This algorithm uses the scikit-learn library:
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    """

    _patched_methods = {
        "fit_transform",
    }

    def __init__(
        self,
        space,
        n_components=2,
        metric_mds=True,
        n_init=1,
        init="classical_mds",
        max_iter=300,
        verbose=0,
        eps=1e-06,
        n_jobs=None,
        random_state=None,
        metric_params=None,
        normalized_stress="auto",
    ):
        self.space = space

        self._set_interop(space)

        super().__init__(
            n_components=n_components,
            metric_mds=metric_mds,
            n_init=n_init,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
            metric=space.metric.dist,
            metric_params=metric_params,
            normalized_stress=normalized_stress,
        )

    def _set_interop(self, space):
        array_repr = isinstance(space, Manifold)
        backend_fix = gs.__name__.endswith("pytorch")

        self._use_sklearn_patches = (
            not array_repr or space.point_ndim > 1 or backend_fix
        )

        if not self._use_sklearn_patches:
            return

        patches = []

        if backend_fix:
            patches.extend(
                [
                    (_mds, "smacof", smacof),
                    (_mds, "ClassicalMDS", ClassicalMDS),
                ]
            )

        if array_repr and space.point_ndim > 1:
            patches.extend(
                [
                    (_mds, "validate_data", validate_data_skip_check_array),
                    (mp, "check_array", check_array_allow_nd),
                ]
            )

        if not array_repr:
            patches.extend(
                [
                    (_mds, "validate_data", validate_data_skip_check_array),
                    (
                        _mds,
                        "pairwise_distances",
                        lambda X, metric: pairwise_dists(X, metric),
                    ),
                ]
            )

        self._sklearn_patches = tuple(patches)
