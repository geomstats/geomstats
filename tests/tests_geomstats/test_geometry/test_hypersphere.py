import logging
import random

import numpy as np
import pytest
from scipy import stats
from scipy.special import ive

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import HypersphereIntrinsicRandomDataGenerator
from geomstats.test_cases.geometry.hypersphere import (
    HypersphereCoordsTransformTestCase,
    HypersphereExtrinsicTestCase,
    HypersphereIntrinsicTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.hypersphere import (
    Hypersphere2ExtrinsicMetricTestData,
    Hypersphere2IntrinsicMetricTestData,
    Hypersphere4ExtrinsicMetricTestData,
    HypersphereCoordsTransformTestData,
    HypersphereExtrinsicMetricTestData,
    HypersphereExtrinsicTestData,
    HypersphereIntrinsicTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        3,
    ],
)
def coords_transform_spaces(request):
    dim = request.param
    request.cls.space_intrinsic = Hypersphere(dim, intrinsic=True)
    request.cls.space = request.cls.space_extrinsic = Hypersphere(dim, intrinsic=False)


@pytest.mark.usefixtures("coords_transform_spaces")
class TestHypersphereCoordsTransform(
    HypersphereCoordsTransformTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereCoordsTransformTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        random.randint(3, 5),
    ],
)
def extrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, intrinsic=False, equip=True)


@pytest.mark.usefixtures("extrinsic_spaces")
class TestHypersphereExtrinsic(
    HypersphereExtrinsicTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereExtrinsicTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
    ],
)
def intrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, intrinsic=True)


@pytest.mark.usefixtures("intrinsic_spaces")
class TestHypersphereIntrinsic(
    HypersphereIntrinsicTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereIntrinsicTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        random.randint(3, 5),
    ],
)
def equipped_extrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, intrinsic=False)


@pytest.mark.usefixtures("equipped_extrinsic_spaces")
class TestHypersphereExtrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereExtrinsicMetricTestData()

    @pytest.mark.random
    def test_sectional_curvature_is_one(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.ones(batch_shape)

        self.assertAllClose(res, expected, atol=atol)


class TestHypersphere2IntrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(2, intrinsic=True, equip=False)
    space.equip_with_metric(HypersphereMetric)

    data_generator = HypersphereIntrinsicRandomDataGenerator(space)
    testing_data = Hypersphere2IntrinsicMetricTestData()

    def test_riemann_tensor_spherical_coords(self, base_point):
        """Test the Riemann tensor on the sphere.

        riemann_tensor[...,i,j,k,l] = R_{ijk}^l
            Riemannian tensor curvature,
            with the contravariant index on the last dimension.

        Note that the base_point is input in spherical coordinates.

        Expected formulas taken from:
        https://digitalcommons.latech.edu/cgi/viewcontent.cgi?
        article=1008&context=mathematics-senior-capstone-papers
        """
        riemann_tensor_ijk_l = self.space.metric.riemann_tensor(base_point)

        theta, _ = base_point[0], base_point[1]
        expected_212_1 = gs.array(gs.sin(theta) ** 2)
        expected_221_1 = gs.array(-(gs.sin(theta) ** 2))
        expected_121_2 = gs.array(1.0)
        expected_112_2 = gs.array(-1.0)
        result_212_1 = riemann_tensor_ijk_l[1, 0, 1, 0]
        result_221_1 = riemann_tensor_ijk_l[1, 1, 0, 0]
        result_121_2 = riemann_tensor_ijk_l[0, 1, 0, 1]
        result_112_2 = riemann_tensor_ijk_l[0, 0, 1, 1]
        self.assertAllClose(expected_212_1, result_212_1)
        self.assertAllClose(expected_221_1, result_221_1)
        self.assertAllClose(expected_121_2, result_121_2)
        self.assertAllClose(expected_112_2, result_112_2)


@pytest.mark.smoke
class TestHypersphere2ExtrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(2, intrinsic=False, equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere2ExtrinsicMetricTestData()

    def test_log_antipodal_raises(self):
        """Log of antipodal points is not well-defined."""
        x = gs.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="antipodal"):
            self.space.metric.log(-x, x)


@pytest.mark.smoke
class TestHypersphere4ExtrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(4, intrinsic=False, equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere4ExtrinsicMetricTestData()

    def test_exp_and_dist_and_projection_to_tangent_space(self, vector, base_point):
        tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)
        exp = self.space.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = self.space.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)


def _circle_angles(point):
    """Extract the angle of points on the circle, in [-pi, pi]."""
    point = np.asarray(point, dtype=np.float64)
    return np.arctan2(point[..., 1], point[..., 0])


def _von_mises_ratio(order, kappa):
    """Compute I_order(kappa) / I_0(kappa) using exponentially scaled Bessels."""
    return ive(order, kappa) / ive(0, kappa)


class TestVonMisesFisherCircle:
    """Regression tests for von Mises-Fisher sampling on the circle.

    On S^1 the von Mises-Fisher distribution is the von Mises distribution,
    whose density with respect to the angle is

        p(theta) = exp(kappa * cos(theta)) / (2 * pi * I_0(kappa)),

    and the generic rejection sampler of the higher-dimensional case does not
    apply. The tolerances below are derived from the theoretical variance of
    each statistic instead of being chosen by hand: the moment and mean
    direction checks use a 5-sigma bound (~6e-7 false-positive rate each), and
    the uniformity check uses a Kolmogorov-Smirnov critical value whose tail
    probability is ~7e-6.
    """

    space = Hypersphere(1)
    n_samples = 20_000

    @pytest.mark.smoke
    @pytest.mark.shape
    @pytest.mark.parametrize("n_samples", [1, 2, 17, 1000])
    def test_shapes_and_unit_norm(self, n_samples):
        point = gs.to_numpy(
            self.space.random_von_mises_fisher(n_samples=n_samples, kappa=10.0)
        )
        expected_shape = (2,) if n_samples == 1 else (n_samples, 2)

        assert point.shape == expected_shape
        assert np.isfinite(point).all()
        assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-10)

    @pytest.mark.random
    @pytest.mark.parametrize("kappa", [0.0, 1e-12, 1e-8])
    def test_vanishing_concentration_is_uniform(self, kappa):
        """kappa == 0 must yield the uniform distribution on the circle.

        This is the exact limit of the von Mises distribution, and it is also
        the regime where the rejection parameters of [BF79]_ degenerate.
        """
        point = gs.to_numpy(
            self.space.random_von_mises_fisher(
                n_samples=self.n_samples, kappa=kappa, mu=gs.array([1.0, 0.0])
            )
        )
        assert np.isfinite(point).all()
        assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-10)

        angles = _circle_angles(point)
        result = stats.kstest(angles, "uniform", args=(-np.pi, 2 * np.pi))
        # 2.5 / sqrt(n) is the Kolmogorov critical value at a tail probability
        # of ~7e-6, i.e. the test is essentially never flaky, while a wrong
        # acceptance criterion produces a statistic an order of magnitude
        # larger (D ~ 6 / sqrt(n) for a sign error in the log branch).
        assert result.statistic < 2.5 / np.sqrt(self.n_samples)

    @pytest.mark.random
    @pytest.mark.parametrize("kappa", [1e-8, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4])
    def test_moments_match_theory(self, kappa):
        """Check the first and second circular moments against I_k / I_0.

        The variances of cos(theta) and cos(2 * theta) are known in closed
        form, which gives a principled tolerance instead of an arbitrary one.
        """
        point = gs.to_numpy(
            self.space.random_von_mises_fisher(
                n_samples=self.n_samples, kappa=kappa, mu=gs.array([1.0, 0.0])
            )
        )
        angles = _circle_angles(point)
        cos_1, cos_2 = np.cos(angles), np.cos(2 * angles)

        expected_1 = _von_mises_ratio(1, kappa)
        expected_2 = _von_mises_ratio(2, kappa)
        var_1 = (1.0 + expected_2) / 2.0 - expected_1**2
        var_2 = (1.0 + _von_mises_ratio(4, kappa)) / 2.0 - expected_2**2

        assert abs(cos_1.mean() - expected_1) < 5.0 * np.sqrt(var_1 / len(angles))
        assert abs(cos_2.mean() - expected_2) < 5.0 * np.sqrt(var_2 / len(angles))
        # E[sin(theta)] = E[sin(2 * theta)] = 0, with variance 1 / 2.
        assert abs(np.sin(angles).mean()) < 5.0 / np.sqrt(2.0 * len(angles))
        assert abs(np.sin(2 * angles).mean()) < 5.0 / np.sqrt(2.0 * len(angles))

    @pytest.mark.random
    @pytest.mark.parametrize(
        "mu", [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, -1.0], [0.6, 0.8]]
    )
    def test_mean_direction(self, mu):
        """The sample mean direction must match `mu`, including after rotation."""
        mu = np.asarray(mu, dtype=np.float64)
        mu = mu / np.linalg.norm(mu)
        kappa = 5.0

        point = gs.to_numpy(
            self.space.random_von_mises_fisher(
                n_samples=self.n_samples, kappa=kappa, mu=gs.array(mu)
            )
        )
        mean_direction = point.mean(axis=0)
        mean_direction = mean_direction / np.linalg.norm(mean_direction)

        # Asymptotic standard error of the mean direction of a von Mises
        # sample is 1 / sqrt(n * kappa * A_1(kappa)).
        standard_error = 1.0 / np.sqrt(
            self.n_samples * kappa * _von_mises_ratio(1, kappa)
        )
        assert np.linalg.norm(mean_direction - mu) < 5.0 * standard_error

    @pytest.mark.smoke
    @pytest.mark.validation
    def test_issue_1334_reproduction(self):
        """The snippet from issue #1334 must now return a point on the circle.

        The report passed an integer NumPy array as `mu`, which only works on
        the numpy and autograd backends: `rotate_points` needs a floating
        point backend array, a pre-existing limitation that also affects
        `dim >= 2`. A float `mu` is used here so the test covers every backend.
        """
        sphere = Hypersphere(dim=1)
        point = gs.to_numpy(sphere.random_von_mises_fisher(mu=gs.array([1.0, 0.0])))

        assert point.shape == (2,)
        assert np.isfinite(point).all()
        assert np.isclose(np.linalg.norm(point), 1.0, atol=1e-10)

    @pytest.mark.validation
    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            self.space.random_von_mises_fisher(n_samples=5, kappa=-1.0)

    @pytest.mark.parametrize("kappa", [0.0, 10.0])
    @pytest.mark.validation
    def test_non_positive_n_samples_raises(self, kappa):
        """n_samples=0 must fail on its own terms, not via max_iter."""
        with pytest.raises(ValueError, match="n_samples"):
            self.space.random_von_mises_fisher(n_samples=0, kappa=kappa)

    @pytest.mark.validation
    def test_max_iter_zero_raises_informative_error(self):
        """A zero budget cannot accept any candidate: fail explicitly."""
        with pytest.raises(ValueError, match="max_iter"):
            self.space.random_von_mises_fisher(n_samples=1, kappa=10.0, max_iter=0)

    @pytest.mark.random
    def test_partial_acceptance_returns_fewer_samples_and_warns(self, caplog):
        """A short batch is returned with a warning, as documented.

        With `max_iter=1` and 1000 candidates, the probability of accepting
        every single one is ~0.66**1000, so the branch is exercised reliably.
        """
        with caplog.at_level(logging.WARNING):
            point = gs.to_numpy(
                self.space.random_von_mises_fisher(
                    n_samples=1000, kappa=10.0, max_iter=1
                )
            )

        assert 0 < point.shape[0] < 1000
        assert point.shape[1] == 2
        assert np.isfinite(point).all()
        assert "Maximum number of iteration" in caplog.text

    @pytest.mark.validation
    def test_exhausted_max_iter_is_never_a_concatenation_error(self):
        """`max_iter=1` may legitimately reject every candidate.

        Whatever the outcome, the caller must either receive a valid point or
        an error that names `max_iter`; an empty `gs.concatenate` is a bug.
        """
        for seed in range(50):
            gs.random.seed(seed)
            try:
                point = self.space.random_von_mises_fisher(
                    n_samples=1, kappa=10.0, max_iter=1
                )
            except ValueError as error:
                assert "max_iter" in str(error)
            else:
                assert gs.to_numpy(point).shape == (2,)

    @pytest.mark.validation
    def test_sampling_is_reproducible(self):
        gs.random.seed(0)
        first = gs.to_numpy(
            self.space.random_von_mises_fisher(n_samples=100, kappa=3.0)
        )
        gs.random.seed(0)
        second = gs.to_numpy(
            self.space.random_von_mises_fisher(n_samples=100, kappa=3.0)
        )
        assert np.array_equal(first, second)

    @pytest.mark.smoke
    @pytest.mark.shape
    def test_other_dimensions_are_unchanged(self):
        """dim >= 2 must keep sampling valid points of the right shape."""
        for dim in [2, 3, 4]:
            space = Hypersphere(dim)
            point = gs.to_numpy(space.random_von_mises_fisher(n_samples=50, kappa=5.0))
            assert point.shape == (50, dim + 1)
            assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-6)
