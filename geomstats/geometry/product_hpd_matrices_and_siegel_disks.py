"""The ProductHPDMatricesAndSiegelDisks manifold.

The ProductHPDMatricesAndSiegelDisks manifold is defined as a
product manifold of the HPD manifold and (n-1) Siegel disks.
The HPD Siegel disks product has a product metric.
The product metric on the HPD Siegel disks product space is the usual
HPD matrices affine-invariant metric (with power affine parameter equal 1)
and Siegel metrics multiplied by constants.
This product manifold can be used to represent Block-Toeplitz HPD matrices.


Lead author: Yann Cabanes.

References
----------
.. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, 2022
.. [Cabanes2021] Yann Cabanes and Frank Nielsen.
    New theoreticla tools in the Siegel space for vectorial
    autoregressive data classification,
    Geometric Science of Information, 2021.
    https://franknielsen.github.io/IG/GSI2021-SiegelLogExpClassification.pdf
.. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

from geomstats.geometry.hpd_matrices import HPDAffineMetric, HPDMatrices
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric  # NOQA
from geomstats.geometry.siegel import Siegel, SiegelMetric


class ProductHPDMatricesAndSiegelDisks(ProductManifold):
    """Class for the ProductHPDMatricesAndSiegelDisks manifold.

    The HPD and Siegel product manifold is a direct product of the HPD manifold
    and (n-1) Siegel disks. Each manifold of the product is a square matrix
    manifold of the same dimension.

    Parameters
    ----------
    n_manifolds : int
        Number of manifolds of the product.
    n : int
        Size of the matrices.
    """

    def __init__(self, n_manifolds, n, **kwargs):
        hpd_matrices = HPDMatrices(n=n)
        hpd_matrices.metric = HPDAffineMetric(n=n, scale=n_manifolds**0.5)
        siegel_disk = Siegel(n=n)
        list_manifolds = [hpd_matrices] + (n_manifolds - 1) * [siegel_disk]
        super(ProductHPDMatricesAndSiegelDisks, self).__init__(
            factors=list_manifolds, **kwargs
        )
        self.shape = (n_manifolds, n, n)
        self.metric = ProductHPDMatricesAndSiegelDisksMetric(
            n_manifolds=n_manifolds, n=n, **kwargs
        )


class ProductHPDMatricesAndSiegelDisksMetric(ProductRiemannianMetric):
    """Class defining the ProductHPDMatricesAndSiegelDisks metric.

    The HPD Siegel disks product metric is a product of the HPD metric
    and (n-1) Siegel metrics, each of them being multiplied by a specific
    constant factor (see [Cabanes2022]_, [Cabanes2021]_ and [JV2016]_).
    This metric comes from a model used to represent multidimensional complex
    stationary centered Gaussian autoregressive times series.
    A multidimensional times series can be seen as a realization of
    a multidimensional complex Gaussian distributions with zero mean,
    a block-Toeplitz HPD covariance matrix and a zero relation matrix.
    The ProductHPDMatricesAndSiegelDisks metric is inspired by information geometry
    on this specific set of Gaussian distributions.

    Parameters
    ----------
    n_manifolds : int
        Number of manifolds of the product.
    n : int
        Size of the matrices.

    References
    ----------
    .. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
        centered Gaussian autoregressive time series machine learning
        in Poincaré and Siegel disks: application for audio and radar
        clutter classification, PhD thesis, 2022
    .. [Cabanes2021] Yann Cabanes and Frank Nielsen.
        New theoreticla tools in the Siegel space for vectorial
        autoregressive data classification,
        Geometric Science of Information, 2021.
        https://franknielsen.github.io/IG/GSI2021-SiegelLogExpClassification.pdf
    .. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
        matrices with Toeplitz structured blocks, 2016.
        https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n_manifolds, n, **kwargs):
        metrics = [HPDAffineMetric(n=n)] + (n_manifolds - 1) * [SiegelMetric(n=n)]
        scales = [
            (n_manifolds - i_manifold) ** 0.5 for i_manifold in range(n_manifolds)
        ]

        super(ProductHPDMatricesAndSiegelDisksMetric, self).__init__(
            metrics=metrics, scales=scales, default_point_type="matrix", **kwargs
        )
        self.shape = (n_manifolds, n, n)
