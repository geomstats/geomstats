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
from geomstats.geometry.product_manifold import ProductManifold, ProductRiemannianMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
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

    def __init__(self, n_manifolds, n, equip=True):
        self.n_manifolds = n_manifolds
        self.n = n

        factors = [HPDMatrices(n=n, equip=False)] + [
            Siegel(n=n, equip=False) for _ in range((n_manifolds - 1))
        ]

        scales = [float(n_manifolds - i_manifold) for i_manifold in range(n_manifolds)]
        factors[0].metric = ScalarProductMetric(HPDAffineMetric(factors[0]), scales[0])
        for factor, scale in zip(factors[1:], scales[1:]):
            factor.metric = ScalarProductMetric(SiegelMetric(factor), scale)

        super().__init__(
            factors=factors,
            point_ndim=3,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ProductHPDMatricesAndSiegelDisksMetric


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
