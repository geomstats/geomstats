"""The ProductPositiveRealsAndComplexPoincareDisks manifold.

The ProductPositiveRealsAndComplexPoincareDisks manifold is defined as a
product manifold of the positive reals manifold and (n-1) complex Poincaré disks.
The positive reals and complex Poincaré disks product has a product metric.
The product metric on the positive reals and complex Poincaré disks product space is
the positive reals metric and (n - 1) complex Poincaré metrics multiplied by constants.
This product manifold can be used to represent Toeplitz HPD matrices.
The ProductPositiveRealsAndComplexPoincareDisks corresponds to
the one-dimensional case of ProductHPDMatricesAndSiegelDisks.
Indeed, PositiveReals is the one-dimensional case of HPDMatrices and
ComplexPoincareDisk is the one-dimensional case of Siegel.
In these one-dimensional manifolds, many simplifications occur compared with
the multidimensional manifolds since matrices commute in dimension 1.


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

from geomstats.geometry.complex_poincare_disk import (
    ComplexPoincareDisk,
    ComplexPoincareDiskMetric,
)
from geomstats.geometry.positive_reals import PositiveReals, PositiveRealsMetric
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric  # NOQA


class ProductPositiveRealsAndComplexPoincareDisks(ProductManifold):
    """Class for the ProductPositiveRealsAndComplexPoincareDisks manifold.

    The positive reals and complex Poincaré disks product manifold is a
    direct product of the positive reals manifold and (n-1) complex Poincaré disks.
    Each manifold of the product is a one-dimensional manifold.

    Parameters
    ----------
    n_manifolds : int
        Number of manifolds of the product.
    """

    def __init__(self, n_manifolds, **kwargs):
        scaled_positive_reals_metric = n_manifolds**0.5 * PositiveRealsMetric()
        positive_reals = PositiveReals(metric=scaled_positive_reals_metric)

        complex_poincare_disk = ComplexPoincareDisk()
        factors = [positive_reals] + (n_manifolds - 1) * [complex_poincare_disk]

        super().__init__(
            factors=factors,
            default_point_type="other",
            metric=ProductPositiveRealsAndComplexPoincareDisksMetric(
                n_manifolds=n_manifolds, **kwargs
            ),
            **kwargs
        )


class ProductPositiveRealsAndComplexPoincareDisksMetric(ProductRiemannianMetric):
    """Class defining the ProductPositiveRealsAndComplexPoincareDisks metric.

    The positive reals and complex Poincaré disks product metric is a product
    of the positive reals metric and (n-1) complex Poincaré metrics, each of them
    being multiplied by a specific constant factor (see [Cabanes2022]_ and [JV2016]_).
    This metric comes from a model used to represent one-dimensional complex
    stationary centered Gaussian autoregressive times series.
    Such a times series can be seen as a realization of
    a multidimensional complex Gaussian distributions with zero mean,
    a Toeplitz HPD covariance matrix and a zero relation matrix.
    The ProductPositiveRealsAndComplexPoincareDisks metric is inspired by
    information geometry on this specific set of Gaussian distributions.

    Parameters
    ----------
    n_manifolds : int
        Number of manifolds of the product.

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

    def __init__(self, n_manifolds, **kwargs):
        scales = [float(n_manifolds - i_manifold) for i_manifold in range(n_manifolds)]
        metrics = [scales[0] * PositiveRealsMetric()] + [
            scale * ComplexPoincareDiskMetric() for scale in scales[1:]
        ]

        super().__init__(metrics=metrics, default_point_type="other", **kwargs)
