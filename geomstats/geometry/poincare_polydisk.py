"""The Poincare polydisk.

The Poincare polydisk is defined as a product manifold of the Hyperbolic space
of dimension 2. The Poincare polydisk has a product metric. The metric on each
space is the natural Poincare metric multiplied by a constant.

Lead author: Yann Cabanes.

References
----------
.. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

import geomstats.backend as gs
from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.base import OpenSet
from geomstats.geometry.hyperboloid import Hyperboloid, HyperboloidMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric  # NOQA


class PoincarePolydisk(ProductManifold):
    r"""Class for the Poincare polydisk.

    The Poincare polydisk is a direct product of n Poincare disks,
    i.e. hyperbolic spaces of dimension 2.

    Parameters
    ----------
    n_disks : int
        Number of disks.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: \'extrinsic\'.
    """

    def __init__(self, n_disks, default_coords_type="extrinsic"):
        self.n_disks = n_disks
        disk = Hyperboloid(2, default_coords_type=default_coords_type)
        list_disks = [
            disk,
        ] * n_disks
        super().__init__(
            factors=list_disks,
            default_point_type="matrix"
        )
        self._metric = PoincarePolydiskMetric(
            n_disks=n_disks, default_coords_type=default_coords_type
        )

    @staticmethod
    def intrinsic_to_extrinsic_coords(point_intrinsic):
        """Convert point from intrinsic to extrensic coordinates.

        Convert the parameterization of a point in the hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., n_disk, dim]
            Point in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., n_disks, dim + 1]
            Point in extrinsic coordinates.
        """
        n_disks = point_intrinsic.shape[1]
        point_extrinsic = gs.stack(
            [
                _Hyperbolic.change_coordinates_system(
                    point_intrinsic[:, i_disk, ...], "intrinsic", "extrinsic"
                )
                for i_disk in range(n_disks)
            ],
            axis=1,
        )
        return point_extrinsic


class PoincarePolydiskMetric(ProductRiemannianMetric):
    r"""Class defining the Poincare polydisk metric.

    The Poincare polydisk metric is a product of n Poincare metrics,
    each of them being multiplied by a specific constant factor (see
    [JV2016]_).

    This metric comes from a model used to represent
    stationary complex autoregressive Gaussian signals.

    Parameters
    ----------
    n_disks : int
        Number of disks.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: \'extrinsic\'.

    References
    ----------
    .. [JV2016] B. Jeuris and R. Vandebril. The Kähler mean of Block-Toeplitz
        matrices with Toeplitz structured blocks, 2016.
        https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n_disks, default_coords_type="extrinsic"):
        self.n_disks = n_disks
        list_metrics = []
        for i_disk in range(n_disks):
            scale_i = (n_disks - i_disk) ** 0.5
            metric_i = HyperboloidMetric(2, default_coords_type, scale_i)
            list_metrics.append(metric_i)
        super().__init__(metrics=list_metrics, default_point_type="matrix")
