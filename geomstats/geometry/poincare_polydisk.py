<<<<<<< HEAD
"""The Poincare Polydisk."""
=======
"""
The Poincare polydisk.

The Poincare polydisk is defined as a product manifold of the Hyperbolic space
of dimension 2. The Poincare polydisk has a product metric. The metric on each
space is the natural Poincare metric multiplied by a constant.

References
----------
The Kahler mean of Block-Toeplitz matrices
with Toeplitz structured blocks
B. Jeuris and R. Vandebril
2016
https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""
>>>>>>> Add an example for the Poincare polydisk.

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hyperbolic import HyperbolicMetric
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric \
    import ProductRiemannianMetric  # NOQA

TOLERANCE = 1e-6


class PoincarePolydisk(ProductManifold):
<<<<<<< HEAD
    """Class for the Poincare polydisk.

    The Poincare polydisk is a direct product of n Poincare disks,
    i.e. hyperbolic spaces of dimension 2.
=======
    """Class defining the Poincare polydisk.

    Class for the Poincare polydisk, which is a direct product
    of n Poincare disks, i.e. hyperbolic spaces of dimension 2.
>>>>>>> Add an example for the Poincare polydisk.
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
<<<<<<< HEAD
        """Convert point from intrinsic to extrensic coordinates.
=======
        """Convert intrinsic to extrinsic coordinates.
>>>>>>> Add an example for the Poincare polydisk.

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
<<<<<<< HEAD
        return gs.array([Hyperbolic._intrinsic_to_extrinsic_coordinates(
            point_intrinsic[i_disks, ...]) for i_disks in range(n_disks)])
=======
        hyperbolic_space = HyperbolicSpace(dimension=2)
        point_extrinsic = gs.vstack(
            [hyperbolic_space.intrinsic_to_extrinsic_coords(
                point_intrinsic=point_intrinsic[i_disks, ...])
                for i_disks in range(n_disks)])
        return point_extrinsic

    def projection_to_tangent_space(self, vector, base_point):
        """Project a vector in the tangent space.

        Project a vector in Minkowski space
        on the tangent space of the Hyperbolic space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dimension + 1]
        base_point : array-like, shape=[n_samples, dimension + 1]

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
        """
        n_disks = base_point.shape[0]
        hyperbolic_space = HyperbolicSpace(dimension=2,
                                           point_type=self.point_type)
        tangent_vec = gs.vstack([HyperbolicSpace.projection_to_tangent_space(
            self=hyperbolic_space,
            vector=vector[i_disks, ...],
            base_point=base_point[i_disks, ...])
            for i_disks in range(n_disks)])
        return tangent_vec
>>>>>>> Add an example for the Poincare polydisk.


class PoincarePolydiskMetric(ProductRiemannianMetric):
    """Class defining the Poincare polydisk metric.

<<<<<<< HEAD
    The Poincare polydisk metric is a product of n Poincare metrics,
    each of them being multiplied by a specific constant factor.

=======
    Class defining the Poincare polydisk metric,
    which is a product of n Poincare metrics,
    each of them being multilplied by a specific constant factor.
>>>>>>> Add an example for the Poincare polydisk.
    This metric come from a model used to represent
    stationary complex signals.

    References
    ----------
    .. [1] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
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
