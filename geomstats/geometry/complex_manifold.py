"""Complex manifold module.

In other words, a topological space that locally resembles
Hermitian space near each point.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.manifold import Manifold


class ComplexManifold(Manifold):
    r"""Class for complex manifolds.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    shape : tuple of int
        Shape of one element of the manifold.
        Optional, default : None.
    metric : ComplexRiemannianMetric
        Metric object to use on the complex manifold.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
        self, dim, shape, metric=None, default_coords_type="intrinsic", **kwargs
    ):
        self.dim = dim
        self.shape = shape
        self.default_coords_type = default_coords_type
        self._metric = metric

    @property
    def metric(self):
        """Riemannian metric associated to the complex manifold."""
        return self._metric

    @metric.setter
    def metric(self, metric):
        if metric is not None:
            if not isinstance(metric, ComplexRiemannianMetric):
                raise ValueError(
                    "The argument must be a ComplexRiemannianMetric object"
                )
            if metric.dim != self.dim:
                metric.dim = self.dim
        self._metric = metric

    def random_tangent_vec(self, base_point, n_samples=1):
        """Generate random tangent vec.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape=[..., dim]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vec at base point.
        """
        if (
            n_samples > 1
            and base_point.ndim > len(self.shape)
            and n_samples != len(base_point)
        ):
            raise ValueError(
                "The number of base points must be the same as the "
                "number of samples, when different from 1."
            )
        vector = gs.cast(
            gs.random.normal(size=(n_samples,) + self.shape),
            dtype=gs.get_default_cdtype(),
        ) + 1j * gs.cast(
            gs.random.normal(size=(n_samples,) + self.shape),
            dtype=gs.get_default_cdtype(),
        )
        return gs.squeeze(self.to_tangent(vector, base_point))
