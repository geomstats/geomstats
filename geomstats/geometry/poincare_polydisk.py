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
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.nfold_manifold import NFoldManifold, NFoldMetric


class PoincarePolydisk(NFoldManifold):
    r"""Class for the Poincare polydisk.

    The Poincare polydisk is a direct product of n Poincare disks,
    i.e. hyperbolic spaces of dimension 2.

    Parameters
    ----------
    n_disks : int
        Number of disks.
    """

    def __init__(self, n_disks, equip=True):
        self.n_disks = n_disks
        super().__init__(base_manifold=Hyperboloid(2), n_copies=n_disks, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PoincarePolydiskMetric

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


class PoincarePolydiskMetric(NFoldMetric):
    r"""Class defining the Poincare polydisk metric.

    The Poincare polydisk metric is a product of n Poincare metrics,
    each of them being multiplied by a specific constant factor (see
    [JV2016]_).

    This metric comes from a model used to represent
    stationary complex autoregressive Gaussian signals.


    References
    ----------
    .. [JV2016] B. Jeuris and R. Vandebril. The Kähler mean of Block-Toeplitz
        matrices with Toeplitz structured blocks, 2016.
        https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, space):
        scales = gs.array(
            [float(space.n_disks - i_disk) for i_disk in range(space.n_disks)]
        )
        super().__init__(space, scales=scales)
