"""The positive real axis and Poincare disks product manifold.

The positive real axis and Poincare disks manifold is defined as a product
manifold of the positive real numbers and the Hyperbolic space of dimension 2.
The positive real axis and Poincare disks has a product metric.
The metric on each space is the natural metric on this space multiplied by a
constant.

Lead author: Yann Cabanes.

References
----------
    .. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, 2022
    .. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
      matrices with Toeplitz structured blocks, 2016.
      https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

import geomstats.backend as gs
from geomstats.geometry.poincare_complex_disk import (
    PoincareComplexDisk,
    PoincareComplexDiskMetric,
)
from geomstats.geometry.positive_reals import PositiveReals, PositiveRealsMetric
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric


class ProductPositiveRealsAndPoincareComplexDisks(ProductManifold):
    """Class for the positive real axis and Poincare disks.

    The positive real axis and poincare disks manifold is
    a direct product of the positive real axis and of
    (n - 1) Poincare disks.
    """

    def __init__(self, n_manifolds):
        """Construct the product manifold."""
        self.n_manifolds = n_manifolds
        positive_real = PositiveReals()
        disk = PoincareComplexDisk()
        list_manifolds = [positive_real,] + [
            disk,
        ] * (n_manifolds - 1)
        super(ProductPositiveRealsAndPoincareComplexDisks, self).__init__(
            manifolds=list_manifolds
        )
        self.metric = ProductPositiveRealsAndPoincareComplexDisksMetric(
            n_manifolds=n_manifolds
        )


class ProductPositiveRealsAndPoincareComplexDisksMetric(ProductRiemannianMetric):
    """Class defining the positive reals and poincare complex disks product metric.

    The Poincare polydisk metric is a product of n Poincare metrics,
    each of them being multiplied by a specific constant factor.

    This metric come from a model used to represent
    stationary complex signals.

    References
    ----------
    .. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
      matrices with Toeplitz structured blocks, 2016.
      https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n_manifolds):
        """Construct the Positive Real and Poincare disks metric."""
        self.n_disks = n_manifolds - 1
        list_metrics = [
            PositiveRealsMetric(scale=n_manifolds**0.5),
        ]
        for i_disk in range(self.n_disks):
            scale_i = (self.n_disks - i_disk) ** 0.5
            metric_i = PoincareComplexDiskMetric(scale=scale_i)
            list_metrics.append(metric_i)
        super(ProductPositiveRealsAndPoincareComplexDisksMetric, self).__init__(
            metrics=list_metrics
        )

    def dist(self, point_a, point_b):
        return gs.real(super().dist(point_a, point_b))

    def squared_dist(self, point_a, point_b):
        return gs.real(super().squared_dist(point_a, point_b))

    def norm(self, vector, base_point):
        return gs.real(super().norm(vector, base_point))

    def squared_norm(self, vector, base_point):
        return gs.real(super().squared_norm(vector, base_point))
