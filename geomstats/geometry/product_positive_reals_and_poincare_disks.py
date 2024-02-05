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
.. [Cabanes_2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, tel-03708515, 2022.
    https://theses.hal.science/tel-03708515
.. [Cabanes_CESAR_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon et
    Jérémie Bigot. Unsupervised Machine Learning for Pathological Radar Clutter
    Clustering: the P-Mean-Shift Algorithm, IEEE, C&ESAR 2019, Rennes, France, 2019.
    https://hal.archives-ouvertes.fr/hal-02875430
.. [Cabanes_RADAR_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon et
    Jérémie Bigot. Non-Supervised High Resolution Doppler Machine Learning for
    Pathological Radar Clutter, IEEE, RADAR 2019, Toulon, France, 2019.
    https://hal.archives-ouvertes.fr/hal-02875415
.. [Cabanes_GSI_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon et
    Jérémie Bigot. Toeplitz Hermitian Positive Definite Matrix Machine Learning
    based on Fisher Metric, IEEE, GSI 2019, Toulouse, France, 2019.
    https://hal.archives-ouvertes.fr/hal-02875403
.. [Le_Brigant_2017] Alice Le Brigant. Probability on the spaces of curves and
    the associated metric spaces using information geometry; radar applications,
    PhD thesis, tel-01635258, 2017.
    https://theses.hal.science/tel-01635258
.. [Jeuris_2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
.. [Yang_2013] Marc Arnaudon, Frédéric Barbaresco and Le Yang. Riemannian Medians
    and Means With Applications to Radar Signal Processing, IEEE, 2013.
"""

from geomstats.geometry.complex_poincare_disk import (
    ComplexPoincareDisk,
    ComplexPoincareDiskMetric,
)
from geomstats.geometry.positive_reals import PositiveReals, PositiveRealsMetric
from geomstats.geometry.product_manifold import ProductManifold, ProductRiemannianMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric


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

    def __init__(self, n_manifolds, equip=True):
        self.n_manifolds = n_manifolds

        factors = [PositiveReals(equip=False)] + [
            ComplexPoincareDisk() for _ in range(n_manifolds - 1)
        ]

        scales = [float(n_manifolds - i_manifold) for i_manifold in range(n_manifolds)]
        factors[0].metric = ScalarProductMetric(
            PositiveRealsMetric(factors[0]), scales[0]
        )
        for factor, scale in zip(factors[1:], scales[1:]):
            factor.metric = ScalarProductMetric(
                ComplexPoincareDiskMetric(factor), scale
            )

        super().__init__(factors=factors, point_ndim=3, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ProductPositiveRealsAndComplexPoincareDisksMetric


class ProductPositiveRealsAndComplexPoincareDisksMetric(ProductRiemannianMetric):
    """Class defining the ProductPositiveRealsAndComplexPoincareDisks metric.

    The positive reals and complex Poincaré disks product metric is a product
    of the positive reals metric and (n-1) complex Poincaré metrics, each of them
    being multiplied by a specific constant factor (see [Cabanes_2022]_,
    [Cabanes_CESAR_2019]_, [Cabanes_RADAR_2019]_, [Cabanes_GSI_2019]_,
    [Le_Brigant_2017], [Jeuris_2016]_ and [Yang_2013]_).
    This metric comes from a model used to represent one-dimensional complex
    stationary centered Gaussian autoregressive times series.
    Such a times series can be seen as a realization of
    a multidimensional complex Gaussian distributions with zero mean,
    a Toeplitz HPD covariance matrix and a zero relation matrix.
    The ProductPositiveRealsAndComplexPoincareDisks metric is inspired by
    information geometry on this specific set of Gaussian distributions.

    References
    ----------
    .. [Cabanes_2022] Yann Cabanes. Multidimensional complex stationary
        centered Gaussian autoregressive time series machine learning
        in Poincaré and Siegel disks: application for audio and radar
        clutter classification, PhD thesis, tel-03708515, 2022.
        https://theses.hal.science/tel-03708515
    .. [Cabanes_CESAR_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon et
        Jérémie Bigot. Unsupervised Machine Learning for Pathological Radar Clutter
        Clustering: the P-Mean-Shift Algorithm, IEEE, C&ESAR 2019, Rennes, France, 2019.
        https://hal.archives-ouvertes.fr/hal-02875430
    .. [Cabanes_RADAR_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon et
        Jérémie Bigot. Non-Supervised High Resolution Doppler Machine Learning for
        Pathological Radar Clutter, IEEE, RADAR 2019, Toulon, France, 2019.
        https://hal.archives-ouvertes.fr/hal-02875415
    .. [Cabanes_GSI_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon et
        Jérémie Bigot. Toeplitz Hermitian Positive Definite Matrix Machine Learning
        based on Fisher Metric, IEEE, GSI 2019, Toulouse, France, 2019.
        https://hal.archives-ouvertes.fr/hal-02875403
    .. [Le_Brigant_2017] Alice Le Brigant. Probability on the spaces of curves and
        the associated metric spaces using information geometry; radar applications,
        PhD thesis, tel-01635258, 2017.
        https://theses.hal.science/tel-01635258
    .. [Jeuris_2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
        matrices with Toeplitz structured blocks, 2016.
        https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    .. [Yang_2013] Marc Arnaudon, Frédéric Barbaresco and Le Yang. Riemannian Medians
        and Means With Applications to Radar Signal Processing, IEEE, 2013.
    """
