import pytest

import geomstats.backend as gs
from geomstats.numerics.finite_differences import forward_difference
from geomstats.test.random import get_random_times
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.nfold_manifold import NFoldManifoldTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.vectorization import get_batch_shape


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
    def test_interpolate(self, point, param, expected, atol):
        points = self.space.interpolate(point)(param)
        self.assertAllClose(points, expected, atol=atol)

    def test_interpolate_vec(self, n_reps, n_times, atol):
        point = self.data_generator.random_point()
        param = get_random_times(n_times)

        expected = self.space.interpolate(point)(param)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point=point,
                    param=param,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_length(self, point, expected, atol):
        length = self.space.length(point)
        self.assertAllClose(length, expected, atol=atol)

    def test_normalize(self, point, expected, atol):
        normalized_point = self.space.normalize(point)
        self.assertAllClose(normalized_point, expected, atol)

    @pytest.mark.random
    def test_normalize_is_unit_length(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        normalized_point = self.space.normalize(point)
        normalize_lengths = self.space.length(normalized_point)
        self.assertAllClose(
            normalize_lengths, gs.ones_like(normalize_lengths), atol=atol
        )


class ReparametrizationBundleTestCase(FiberBundleTestCase):
    @pytest.mark.random
    def test_tangent_vector_projections_orthogonality_with_metric(self, n_points, atol):
        """Test horizontal and vertical projections.

        Check that horizontal and vertical parts of any tangent
        vector are orthogonal with respect to the SRVMetric inner
        product.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        tangent_vec_hor = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )
        tangent_vec_ver = self.total_space.fiber_bundle.vertical_projection(
            tangent_vec, base_point
        )

        res = self.total_space.metric.inner_product(
            tangent_vec_hor, tangent_vec_ver, base_point
        )
        expected_shape = get_batch_shape(self.total_space.point_ndim, base_point)
        expected = gs.zeros(expected_shape)
        self.assertAllClose(res, expected, atol=atol)


class ElasticMetricTestCase(PullbackDiffeoMetricTestCase):
    @pytest.mark.random
    def test_dist_against_no_transform(self, n_points, atol):
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        dist = self.space.metric.dist(point_a, point_b)
        dist_ = elastic_distance(
            self.space,
            point_a,
            point_b,
            a=self.space.metric.a,
            b=self.space.metric.b,
        )

        self.assertAllClose(dist, dist_, atol=atol)
