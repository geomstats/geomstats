"""The manifold of positive reals.

This manifold is a particular case in dimension 1
of the SPD matrices endowed with the SPD affine-invariant metric
with a power affine coefficient equal to one.
This manifold is used as part of a product manifold involving
complex Poincare disks to classify complex stationary centered
Gaussian autoregressive time series in [Cabanes2022]_, [Cabanes_CESAR_2019]_,
[Cabanes_GSI_2019]_ and [Cabanes_RADAR_2019]_.
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

import math

import geomstats.backend as gs
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.open_interval import OpenInterval, OpenIntervalEuclideanMetric
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out


class LogDiffeo(Diffeo):
    r"""Logarithm diffeomorphism from the positive reals to the reals.

    .. math::

        \log: \mathbb{R}_{>0} \rightarrow \mathbb{R}, \quad x \mapsto \log x

    with inverse the exponential function and differentials

    .. math::

        d\log_x(u) = \frac{u}{x}, \quad d\exp_z(w) = e^z w
    """

    def __call__(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., 1]
            Image point.
        """
        return gs.log(base_point)

    def inverse(self, image_point):
        """Inverse diffeomorphism at image point.

        Parameters
        ----------
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., 1]
            Base point.
        """
        return gs.exp(image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        return tangent_vec / base_point

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Inverse tangent diffeomorphism at image point.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., 1]
            Image point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        return base_point * image_tangent_vec


class PowerDiffeo(Diffeo):
    r"""Power diffeomorphism from the positive reals to a half-line.

    .. math::

        x \mapsto x^p / p

    Parameters
    ----------
    power : float
        Non-zero power.
    """

    def __init__(self, power):
        super().__init__()
        self.power = power

    def __call__(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., 1]
            Image point.
        """
        return base_point**self.power / self.power

    def inverse(self, image_point):
        """Inverse diffeomorphism at image point.

        Parameters
        ----------
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., 1]
            Base point.
        """
        return (self.power * image_point) ** (1 / self.power)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        return base_point ** (self.power - 1) * tangent_vec

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Inverse tangent diffeomorphism at image point.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., 1]
            Image point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        return image_tangent_vec / base_point ** (self.power - 1)


class PositiveReals(OpenInterval):
    """Class for the manifold of positive reals.

    The real positive axis endowed with the Information geometry metric.
    """

    def __init__(self, equip=True):
        super().__init__(lower_bound=0.0, upper_bound=math.inf, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return LogMetric

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


class LogMetric(RiemannianMetric):
    """Class for the logarithmic metric on the positive reals.

    It is a particular case in dimension 1
    of the SPD affine-invariant metric
    with a power affine coefficient equal to one.
    The space is one-dimensional, so the metric is flat.
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

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        The only entry is :math:`\partial_x g_{xx} = -2 / x^3`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        metric_derivative : array-like, shape=[..., 1, 1, 1]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        return gs.reshape(-2 / base_point**3, base_point.shape + (1, 1))

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        The only symbol is :math:`\Gamma^x_{xx} = -1 / x`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., 1, 1, 1]
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """
        return gs.reshape(-1 / base_point, base_point.shape + (1, 1))

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        The only entry is :math:`\partial_x \Gamma^x_{xx} = 1 / x^2`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., 1, 1, 1, 1]
            Jacobian of the Christoffel symbols.
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`
        """
        return gs.reshape(1 / base_point**2, base_point.shape + (1, 1, 1))

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

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between `base_point` and `end_point`, or
        alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`. Denoting `tangent_vec` by :math:`v`, `base_point` by
        :math:`x` and `end_point` by :math:`y`, the parallel transport is:

        .. math::
            v' = v \frac{y}{x}

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., 1]
            Point to transport from.
        direction : array-like, shape=[..., 1]
            Tangent vector at base point, initial speed of the geodesic along
            which the parallel transport is computed. Unused if `end_point` is
            given.
        end_point : array-like, shape=[..., 1]
            Point to transport to.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., 1]
            Transported tangent vector at `end_point`.
        """
        if end_point is None:
            end_point = self.exp(direction, base_point)

        return tangent_vec * end_point / base_point

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

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image. The positive reals are isometric to the real
        line through the logarithm, so it is infinite.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class PowerMetric(PullbackDiffeoMetric):
    r"""Class for the power-Euclidean metric on the positive reals.

    It is the pullback of the Euclidean metric on a half-line through
    :class:`PowerDiffeo`, i.e. :math:`g_x = x^{2p - 2}`, and a particular
    case in dimension 1 of the SPD power-Euclidean metric.
    The limit :math:`p \to 0` is the logarithmic metric.
    The space is one-dimensional, so the metric is flat.

    Parameters
    ----------
    space : PositiveReals
        Manifold.
    power : float
        Non-zero power.
    """

    def __init__(self, space, power):
        if power == 0:
            raise ValueError("The power must be non-zero. Use `LogMetric` instead.")
        if power > 0:
            image_space = PositiveReals(equip=False)
        else:
            image_space = OpenInterval(-math.inf, 0.0, equip=False)
        image_space.equip_with_metric(OpenIntervalEuclideanMetric)

        super().__init__(
            space=space, diffeo=PowerDiffeo(power), image_space=image_space
        )
        self.power = power

    def metric_matrix(self, base_point):
        r"""Compute the inner product matrix at base point.

        The only entry is :math:`g_{xx} = x^{2p - 2}`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., 1, 1]
            Inner product matrix.
        """
        return gs.expand_dims(base_point ** (2 * self.power - 2), axis=-1)

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        The only entry is :math:`\partial_x g_{xx} = (2p - 2) x^{2p - 3}`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        metric_derivative : array-like, shape=[..., 1, 1, 1]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        derivative = (2 * self.power - 2) * base_point ** (2 * self.power - 3)
        return gs.reshape(derivative, base_point.shape + (1, 1))

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        The only symbol is :math:`\Gamma^x_{xx} = (p - 1) / x`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., 1, 1, 1]
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """
        return gs.reshape((self.power - 1) / base_point, base_point.shape + (1, 1))

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        The only entry is :math:`\partial_x \Gamma^x_{xx} = (1 - p) / x^2`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., 1, 1, 1, 1]
            Jacobian of the Christoffel symbols.
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`
        """
        return gs.reshape(
            (1 - self.power) / base_point**2, base_point.shape + (1, 1, 1)
        )
