"""The manifold of positive reals.

This manifold is a particular case in dimension 1
of the SPD matrices endowed with the SPD affine-invariant metric
with a power affine coefficient equal to one.
This manifold is used as part of a product manifold involving
complex Poincare disks to classify complex stationary centered
Gaussian autoregressive time series in [Cabanes2022], [Cabanes_CESAR_2019],
[Cabanes_GSI_2019] and [Cabanes_RADAR_2019].
This product manifold is called ProductPositiveRealsAndComplexPoincareDisks
in geomstats. It is a product manifold of complex type, therefore the manifold
PositiveReals is compatible with complex input points and vectors even if the
input values should be reals.


Lead author: Yann Cabanes.

References
----------
.. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, 2022
.. [Cabanes_CESAR_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon,
    Jérémie Bigot. Unsupervised Machine Learning for Pathological Radar
    Clutter Clustering: the P-Mean-Shift Algorithm, C&ESAR 2019, Nov 2019,
    Rennes, France. hal-02875430
    https://hal.archives-ouvertes.fr/hal-02875430/document
.. [Cabanes_GSI_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon,
    Jérémie Bigot. Toeplitz Hermitian Positive Definite Matrix Machine Learning
    based on Fisher Metric. Geometric Science of Information, 2019,
    10.1007/978-3-030-26980-7_27. hal-02875403
    https://hal.archives-ouvertes.fr/hal-02875403/document
.. [Cabanes_RADAR_2019] Yann Cabanes, Frédéric Barbaresco, Marc Arnaudon,
    Jérémie Bigot. Non-Supervised High Resolution Doppler Machine Learning
    for Pathological Radar Clutter. RADAR 2019, Sep 2020, Toulon, France.
    10.1109/RADAR41533.2019.171295. hal-02875415
    https://hal.archives-ouvertes.fr/hal-02875415/document
.. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PositiveReals(OpenSet):
    """Class for the manifold of positive reals.

    The real positive axis endowed with the Information geometry metric.
    """

    def __init__(self, equip=True):
        super().__init__(
            dim=1,
            embedding_space=Euclidean(1),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PositiveRealsMetric

    @staticmethod
    def belongs(point, atol=gs.atol):
        """Check if a point is a positive real.

        Evaluate if a point belongs to the positive real axis,
        i.e. evaluate if its norm is lower than one.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Points to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...]
            Boolean denoting if point is a positive real.
        """
        if point.shape[-1] != 1:
            return gs.zeros(point.shape[:-1], dtype=bool)

        is_real = gs.all(gs.abs(gs.imag(point)) < atol, axis=(-1,))
        is_positive = gs.all(gs.real(point) > 0, axis=(-1,))

        res = gs.logical_and(is_positive, is_real)
        if not gs.is_array(res):
            return gs.array(res)

        return res

    @staticmethod
    def projection(point, atol=gs.atol):
        """Project a point on the positive reals.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point in ambient manifold.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        projected : array-like, shape=[..., 1]
            Projected point.
        """
        return gs.where(point <= 0, atol, point)

    @staticmethod
    def random_point(n_samples=1, bound=1.0, atol=gs.atol):
        """Sample in the positive reals.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        samples : array-like, shape=[..., 1]
            Points sampled in the positive reals.
        """
        size = (n_samples, 1) if n_samples != 1 else (1,)
        return (bound - atol) * gs.random.rand(*size) + atol


class PositiveRealsMetric(RiemannianMetric):
    """Class for the positive reals metric.

    It is a particular case in dimension 1
    of the SPD affine-invariant metric
    with a power affine coefficient equal to one.
    """

    def metric_matrix(self, base_point):
        """Compute the inner product matrix at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., 1, 1]
            Inner product matrix.
        """
        inner_prod_mat = 1 / base_point**2
        return gs.expand_dims(inner_prod_mat, axis=-1)

    @staticmethod
    def exp(tangent_vec, base_point):
        """Compute the positive reals exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., 1]
            Riemannian exponential.
        """
        return base_point * gs.exp(tangent_vec / base_point)

    @staticmethod
    def log(point, base_point):
        """Compute the positive reals logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the positive reals metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., 1]
            Riemannian logarithm.
        """
        return base_point * gs.log(point / base_point)

    def squared_dist(self, point_a, point_b):
        """Compute the positive reals squared distance.

        Compute the Riemannian squared distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point.
        point_b : array-like, shape=[..., 1]
            Point.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            Riemannian squared distance.
        """
        ratio = gs.squeeze(point_b / point_a, axis=-1)
        return (gs.log(ratio)) ** 2

    def dist(self, point_a, point_b):
        """Compute the positive reals distance.

        Compute the Riemannian distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point.
        point_b : array-like, shape=[..., 1]
            Point.

        Returns
        -------
        dist : array-like, shape=[...]
            Riemannian distance.
        """
        ratio = gs.squeeze(point_b / point_a, axis=-1)
        return gs.abs(gs.log(ratio))
