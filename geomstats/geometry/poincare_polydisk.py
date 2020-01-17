"""
The Poincare polydisk
"""

from geomstats.geometry.hyperbolic_space import HyperbolicMetric
from geomstats.geometry.hyperbolic_space import HyperbolicSpace
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric \
    import ProductRiemannianMetric  # NOQA


class PoincarePolydisk(ProductManifold):
    """
    Class for the Poincare polydisk, which is a direct product
    of n Poincare disks, i.e. hyperbolic spaces of dimension 2.
    """
    def __init__(self, n_disks):
        self.n_disks = n_disks
        disk = HyperbolicSpace(dimension=2)
        list_disks = [disk, ] * n_disks
        super(PoincarePolydisk, self).__init__(
            manifolds=list_disks)
        self.metric = PoincarePolydiskMetric(n_disks)


class PoincarePolydiskMetric(ProductRiemannianMetric):
    """
    Class defining the Poincare polydisk metric,
    which is a product of n Poincare metrics,
    each of them being multilplied by a specific constant factor.
    This metric come from a model used to represent
    stationary complex signals.

    References
    ----------
    The Kahler mean of Block-Toeplitz matrices
    with Toeplitz structured blocks
    B. Jeuris and R. Vandebril
    2016
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n_disks):
        self.n_disks = n_disks
        list_metrics = []
        for i_disk in range(n_disks):
            scale_i = (n_disks - i_disk) ** 0.5
            metric_i = HyperbolicMetric(dimension=2, scale=scale_i)
            list_metrics.append(metric_i)
        super(PoincarePolydiskMetric, self).__init__(
                metrics=list_metrics)
