"""The Poincare Polydisk."""

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hyperbolic import HyperbolicMetric
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric \
    import ProductRiemannianMetric  # NOQA


class PoincarePolydisk(ProductManifold):
    """Class for the Poincare polydisk.

    The Poincare polydisk is a direct product of n Poincare disks,
    i.e. hyperbolic spaces of dimension 2.
    """

    def __init__(self, n_disks, point_type='ball'):
        self.n_disks = n_disks
        self.point_type = point_type
        disk = Hyperbolic(dimension=2, point_type=point_type)
        list_disks = [disk, ] * n_disks
        super(PoincarePolydisk, self).__init__(
            manifolds=list_disks)
        self.metric = PoincarePolydiskMetric(n_disks=n_disks,
                                             point_type=point_type)

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert point from intrinsic to extrensic coordinates.

        Convert the parameterization of a point on the Hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_diskx, n_samples, dimension]

        Returns
        -------
        point_extrinsic : array-like, shape=[n_disks, n_samples, dimension + 1]
        """
        n_disks = point_intrinsic.shape[0]
        return gs.array([Hyperbolic._intrinsic_to_extrinsic_coordinates(
            point_intrinsic[i_disks, ...]) for i_disks in range(n_disks)])


class PoincarePolydiskMetric(ProductRiemannianMetric):
    """Class defining the Poincare polydisk metric.

    The Poincare polydisk metric is a product of n Poincare metrics,
    each of them being multiplied by a specific constant factor (see
    [JV2016]_).

    This metric comes from a model used to represent stationary complex
    signals.

    References
    ----------
    .. [JV2016] B. Jeuris and R. Vandebril. The Kähler mean of Block-Toeplitz
      matrices with Toeplitz structured blocks, 2016.
      https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n_disks, point_type='ball'):
        self.n_disks = n_disks
        self.point_type = point_type
        list_metrics = []
        for i_disk in range(n_disks):
            scale_i = (n_disks - i_disk) ** 0.5
            metric_i = HyperbolicMetric(dimension=2,
                                        point_type=point_type,
                                        scale=scale_i)
            list_metrics.append(metric_i)
        super(PoincarePolydiskMetric, self).__init__(
            metrics=list_metrics)
