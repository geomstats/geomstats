import geomstats.backend as gs
from geomstats.numerics.finite_differences import forward_difference
from geomstats.test_cases.geometry.nfold_manifold import NFoldManifoldTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase


def elastic_distance(space, point_a, point_b, a=1.0, b=0.5):
    r"""Elastic distance between two points.

    Does not use SRV transform. Implements theorem 2.1 of [BCKKNP2022]_:

    .. math::

        \operatorname{dist}_{a, b}\left(c_1, c_2\right)=
        2 b \sqrt{\ell_{c_1}+\ell_{c_2}-2 \int_D \sqrt{\left|c_1^{\prime}
        \right|\left|c_2^{\prime}\right|} \cos \left(\frac{a}{2 b} \theta\right) d u}

    with

    .. math::

        \theta(u)=\min \left(\cos ^{-1}\left(R\left(c_1\right)
        \cdot R\left(c_2\right) /\left|R\left(c_1\right) \|
        R\left(c_2\right)\right|\right), \frac{2 b \pi}{a}\right)

    Parameters
    ----------
    space : DiscreteCurvesStartingAtOrigin
        Unequipped space.
    point_a : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
        Point.
    point_b : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
        Point.
    a : float
        Bending parameter.
    b : float
        Stretching parameter.

    Returns
    -------
    dist : array-like, shape=[...,]
        Distance.

    References
    ----------
    .. [BCKKNP2022] Martin Bauer, Nicolas Charon, Eric Klassen, Sebastian Kurtek,
        Tom Needham, and Thomas Pierron.
        “Elastic Metrics on Spaces of Euclidean Curves: Theory and Algorithms.”
        arXiv, September 20, 2022. https://doi.org/10.48550/arXiv.2209.09862.
    """
    lambda_ = a / (2 * b)
    k_sampling_points = point_a.shape[-2] + 1

    delta = 1 / (k_sampling_points - 1)

    velocity_a = forward_difference(space.insert_origin(point_a), axis=-2)
    velocity_b = forward_difference(space.insert_origin(point_b), axis=-2)

    length_a = space.length(point_a)
    length_b = space.length(point_b)

    velocity_a_norm = gs.linalg.norm(velocity_a, axis=-1)
    velocity_b_norm = gs.linalg.norm(velocity_b, axis=-1)

    aux_theta = gs.arccos(
        gs.dot(velocity_a, velocity_b) / (velocity_a_norm * velocity_b_norm)
    )
    max_value = gs.pi / lambda_

    theta = gs.where(
        aux_theta > max_value,
        max_value,
        aux_theta,
    )

    integral = gs.sum(
        gs.sqrt(velocity_a_norm * velocity_b_norm) * gs.cos(lambda_ * theta) * delta,
        axis=-1,
    )

    return 2 * b * gs.sqrt(length_a + length_b - 2 * integral)


class DiscreteCurvesStartingAtOriginTestCase(NFoldManifoldTestCase):
    def test_interpolate(self, point, time, expected, atol=gs.atol):
        """Check interpolation.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        time : array-like, shape=[n_times]
            Parameters at which to evaluate the interpolated curve.
        expected : array-like, shape=[..., n_times, ambient_dim]
            Expected points of the interpolated curve.
        atol : float
            Absolute tolerance.
        """
        points = self.space.interpolate(point)(time)
        self.assertAllClose(points, expected, atol=atol)

    def test_length(self, point, expected, atol=gs.atol):
        """Check length.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        expected : array-like, shape=[...]
            Expected length.
        atol : float
            Absolute tolerance.
        """
        length = self.space.length(point)
        self.assertAllClose(length, expected, atol=atol)

    def test_normalize(self, point, expected, atol=gs.atol):
        """Check normalization.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        expected : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Expected normalized curve.
        atol : float
            Absolute tolerance.
        """
        normalized_point = self.space.normalize(point)
        self.assertAllClose(normalized_point, expected, atol=atol)

    def test_interpolate_at_sampling_points(self, point, atol=gs.atol):
        """Check interpolation passes through the sampling points.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        atol : float
            Absolute tolerance.
        """
        sampling_times = gs.linspace(0.0, 1.0, self.space.k_sampling_points)

        res = self.space.interpolate(point)(sampling_times)
        expected = self.space.insert_origin(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_normalize_is_unit_length(self, point, atol=gs.atol):
        """Check normalized curves have unit length.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        atol : float
            Absolute tolerance.
        """
        normalized_point = self.space.normalize(point)
        normalize_lengths = self.space.length(normalized_point)
        self.assertAllClose(
            normalize_lengths, gs.ones_like(normalize_lengths), atol=atol
        )


class ElasticMetricTestCase(PullbackDiffeoMetricTestCase):
    def test_dist_against_no_transform(self, point_a, point_b, atol=gs.atol):
        """Check distance against a closed form not using the transform.

        Parameters
        ----------
        point_a : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        point_b : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        atol : float
            Absolute tolerance.
        """
        dist = self.space.metric.dist(point_a, point_b)
        dist_ = elastic_distance(
            self.space,
            point_a,
            point_b,
            a=self.space.metric.a,
            b=self.space.metric.b,
        )

        self.assertAllClose(dist, dist_, atol=atol)
