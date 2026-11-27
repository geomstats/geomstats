import logging
import random

import numpy as np
import pytest
from scipy import stats
from scipy.special import ive

import geomstats.backend as gs
from geomstats.geometry.hypersphere import (
    AngleToExtrinsicDiffeo,
    CircleAngles,
    Hypersphere,
    HypersphereMetric,
    HypersphereRiemannianNormal,
    HypersphereUniform,
    SphereSphericalCoordinates,
    SphericalToExtrinsicDiffeo,
    VonMisesFisher,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import (
    DiffeomorphicManifoldTestCase,
    LevelSetTestCase,
)
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.hypersphere import (
    DistributionTestCase,
    HypersphereRiemannianNormalTestCase,
    VonMisesFisherTestCase,
)
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)

from .data.base import LevelSetTestData
from .data.diffeo import DiffeoTestData
from .data.hypersphere import (
    CircleAnglesMetricCmpTestData,
    CircleAnglesMetricTestData,
    DistributionTestData,
    Hypersphere2MetricTestData,
    Hypersphere4MetricTestData,
    HypersphereMetricTestData,
    HypersphereRiemannianNormalFrechetMeanTestData,
    SphereSphericalCoordinatesMetricCmpTestData,
    SphereSphericalCoordinatesMetricTestData,
    SphereSphericalCoordinatesTestData,
    VonMisesFisherKappaTestData,
    VonMisesFisherTestData,
)
from .data.manifold import ManifoldTestData


class TestAngleToExtrinsicDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    space = CircleAngles(equip=False)
    image_space = Hypersphere(1, equip=False)
    diffeo = AngleToExtrinsicDiffeo()
    testing_data = DiffeoTestData()


class TestSphericalToExtrinsicDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    space = SphereSphericalCoordinates(equip=False)
    image_space = Hypersphere(2, equip=False)
    diffeo = SphericalToExtrinsicDiffeo()
    testing_data = DiffeoTestData()


class TestHypersphere(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Hypersphere(random.randint(1, 5), equip=False)
    testing_data = LevelSetTestData()


@pytest.mark.smoke
class TestHypersphere2Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = space = Hypersphere(2, equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere2MetricTestData()

    def test_log_antipodal_raises(self):
        """Log of antipodal points is not well-defined."""
        x = gs.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="antipodal"):
            self.space.metric.log(-x, x)


@pytest.mark.smoke
class TestHypersphere4Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = space = Hypersphere(4, equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere4MetricTestData()

    def test_exp_and_dist_and_projection_to_tangent_space(self, vector, base_point):
        tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)
        exp = self.space.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = self.space.metric.dist(base_point, exp)
        expected = gs.linalg.vector_norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)


class TestHypersphereMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = Hypersphere(random.randint(2, 5))
    testing_data = HypersphereMetricTestData()

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


class TestCircleAngles(ManifoldTestCase, metaclass=DataBasedParametrizer):
    space = CircleAngles(equip=False)
    testing_data = ManifoldTestData()


class TestCircleAnglesMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = CircleAngles()
    testing_data = CircleAnglesMetricTestData()


@pytest.mark.redundant
class TestCircleAnglesMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    space = CircleAngles()
    other_space = Hypersphere(1)
    diffeo = AngleToExtrinsicDiffeo()
    testing_data = CircleAnglesMetricCmpTestData()


class TestSphereSphericalCoordinates(
    DiffeomorphicManifoldTestCase, metaclass=DataBasedParametrizer
):
    space = SphereSphericalCoordinates(equip=False)
    testing_data = SphereSphericalCoordinatesTestData()


class TestSphereSphericalCoordinatesMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SphereSphericalCoordinates()
    testing_data = SphereSphericalCoordinatesMetricTestData()

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


@pytest.mark.redundant
class TestSphereSphericalCoordinatesMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    space = SphereSphericalCoordinates()
    other_space = Hypersphere(2)
    diffeo = SphericalToExtrinsicDiffeo()
    testing_data = SphereSphericalCoordinatesMetricCmpTestData()


class TestHypersphereUniform(DistributionTestCase, metaclass=DataBasedParametrizer):
    distribution = HypersphereUniform(Hypersphere(random.randint(1, 5), equip=False))
    testing_data = DistributionTestData()


def _random_kappa():
    return gs.random.uniform(low=1.0, high=10000.0, size=(1,))[0]


@pytest.fixture(
    scope="class",
    params=[
        (1, True),
        (1, False),
        (2, True),
        (2, False),
        (random.randint(3, 5), True),
        (random.randint(3, 5), False),
    ],
)
def von_mises_fisher_distributions(request):
    dim, random_mu = request.param
    space = Hypersphere(dim, equip=False)
    mu = space.random_point() if random_mu else None
    request.cls.distribution = VonMisesFisher(space, mu=mu, kappa=_random_kappa())


@pytest.mark.usefixtures("von_mises_fisher_distributions")
class TestVonMisesFisher(VonMisesFisherTestCase, metaclass=DataBasedParametrizer):
    testing_data = VonMisesFisherTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        random.randint(3, 5),
    ],
)
def von_mises_fisher_kappa_distributions(request):
    dim = request.param
    request.cls.distribution = VonMisesFisher(Hypersphere(dim, equip=False), kappa=1.0)


@pytest.mark.usefixtures("von_mises_fisher_kappa_distributions")
class TestVonMisesFisherKappa(VonMisesFisherTestCase, metaclass=DataBasedParametrizer):
    testing_data = VonMisesFisherKappaTestData()


def _random_precision(dim, precision_type):
    if precision_type is None:
        return None

    precision = gs.random.uniform(low=1.0, high=10.0, size=(1,))[0]
    if precision_type is float:
        return precision

    return precision * gs.eye(dim)


@pytest.fixture(
    scope="class",
    params=[
        (dim, random_mean, precision_type)
        for dim in [1, 2, random.randint(3, 5)]
        for random_mean, precision_type in zip(
            [True, False, False],
            [None, float, "array"],
        )
    ],
)
def riemannian_normal_distributions(request):
    dim, random_mean, precision_type = request.param
    space = Hypersphere(dim)
    mean = space.random_point() if random_mean else None
    request.cls.distribution = HypersphereRiemannianNormal(
        space, mean=mean, precision=_random_precision(dim, precision_type)
    )


@pytest.mark.usefixtures("riemannian_normal_distributions")
class TestHypersphereRiemannianNormal(
    HypersphereRiemannianNormalTestCase, metaclass=DataBasedParametrizer
):
    testing_data = DistributionTestData()


@pytest.fixture(
    scope="class",
    params=[
        (dim, random_mean)
        for dim in [1, 2, random.randint(3, 5)]
        for random_mean in [True, False]
    ],
)
def riemannian_normal_frechet_mean_distributions(request):
    dim, random_mean = request.param
    space = Hypersphere(dim)
    mean = space.random_point() if random_mean else None
    request.cls.distribution = HypersphereRiemannianNormal(
        space, mean=mean, precision=_random_precision(dim, "array")
    )


@pytest.mark.usefixtures("riemannian_normal_frechet_mean_distributions")
class TestHypersphereRiemannianNormalFrechetMean(
    HypersphereRiemannianNormalTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereRiemannianNormalFrechetMeanTestData()


def _circle_angles(point):
    """Extract the angle of points on the circle, in [-pi, pi]."""
    point = np.asarray(point, dtype=np.float64)
    return np.arctan2(point[..., 1], point[..., 0])


def _von_mises_ratio(order, kappa):
    """Compute I_order(kappa) / I_0(kappa) using exponentially scaled Bessels."""
    return ive(order, kappa) / ive(0, kappa)


class TestVonMisesFisherCircle:
    """Regression tests for von Mises-Fisher sampling on the circle."""

    space = Hypersphere(1)
    n_samples = 20_000

    @pytest.mark.smoke
    def test_distribution_factory_supports_circle(self):
        distribution = VonMisesFisher(self.space, kappa=1.0)
        point = gs.to_numpy(distribution.sample())

        assert point.shape == (2,)
        assert np.isfinite(point).all()
        assert np.isclose(np.linalg.norm(point), 1.0, atol=1e-10)

    @pytest.mark.smoke
    @pytest.mark.shape
    @pytest.mark.parametrize("n_samples", [1, 2, 17, 1000])
    def test_shapes_and_unit_norm(self, n_samples):
        gs.random.seed(0)
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
        gs.random.seed(0)
        point = gs.to_numpy(
            self.space.random_von_mises_fisher(
                n_samples=self.n_samples, kappa=kappa, mu=gs.array([1.0, 0.0])
            )
        )
        assert np.isfinite(point).all()
        assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-10)

        angles = _circle_angles(point)
        result = stats.kstest(angles, "uniform", args=(-np.pi, 2 * np.pi))
        assert result.statistic < 2.5 / np.sqrt(self.n_samples)

    @pytest.mark.random
    @pytest.mark.parametrize("kappa", [1e-8, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4])
    def test_moments_match_theory(self, kappa):
        gs.random.seed(0)
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
        assert abs(np.sin(angles).mean()) < 5.0 / np.sqrt(2.0 * len(angles))
        assert abs(np.sin(2 * angles).mean()) < 5.0 / np.sqrt(2.0 * len(angles))

    @pytest.mark.random
    @pytest.mark.parametrize(
        "mu", [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, -1.0], [0.6, 0.8]]
    )
    def test_mean_direction(self, mu):
        gs.random.seed(0)
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

        standard_error = 1.0 / np.sqrt(
            self.n_samples * kappa * _von_mises_ratio(1, kappa)
        )
        assert np.linalg.norm(mean_direction - mu) < 5.0 * standard_error

    @pytest.mark.smoke
    @pytest.mark.validation
    def test_issue_1334_reproduction(self):
        sphere = Hypersphere(dim=1)
        point = gs.to_numpy(sphere.random_von_mises_fisher(mu=gs.array([1.0, 0.0])))

        assert point.shape == (2,)
        assert np.isfinite(point).all()
        assert np.isclose(np.linalg.norm(point), 1.0, atol=1e-10)

    @pytest.mark.validation
    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            self.space.random_von_mises_fisher(n_samples=5, kappa=-1.0)

    @pytest.mark.validation
    @pytest.mark.parametrize("kappa", [np.nan, np.inf, -np.inf])
    def test_non_finite_kappa_raises(self, kappa):
        with pytest.raises(ValueError, match="finite"):
            self.space.random_von_mises_fisher(n_samples=5, kappa=kappa)

    @pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
    @pytest.mark.validation
    def test_invalid_n_samples_raises(self, n_samples):
        with pytest.raises(ValueError, match="positive integer"):
            self.space.random_von_mises_fisher(n_samples=n_samples, kappa=1.0)

    @pytest.mark.validation
    def test_max_iter_zero_raises_informative_error(self):
        with pytest.raises(ValueError, match="max_iter"):
            self.space.random_von_mises_fisher(n_samples=1, kappa=10.0, max_iter=0)

    @pytest.mark.parametrize("max_iter", [-1, 1.5, True])
    @pytest.mark.validation
    def test_invalid_max_iter_raises(self, max_iter):
        with pytest.raises(ValueError, match="max_iter"):
            self.space.random_von_mises_fisher(
                n_samples=1, kappa=1.0, max_iter=max_iter
            )

    @pytest.mark.smoke
    def test_uniform_sampling_does_not_use_max_iter(self):
        point = gs.to_numpy(
            self.space.random_von_mises_fisher(n_samples=5, kappa=0.0, max_iter=0)
        )

        assert point.shape == (5, 2)
        assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-10)

    @pytest.mark.random
    def test_partial_acceptance_returns_fewer_samples_and_warns(self, caplog):
        gs.random.seed(0)
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
    def test_smallest_positive_concentration_is_finite(self):
        kappa = np.nextafter(0.0, 1.0)
        point = gs.to_numpy(
            self.space.random_von_mises_fisher(n_samples=100, kappa=kappa)
        )

        assert point.shape == (100, 2)
        assert np.isfinite(point).all()
        assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-10)

    @pytest.mark.smoke
    @pytest.mark.parametrize("kappa", [1e4, 1e12, 1e16, 1e100, 1e200, 1e308])
    def test_large_concentration_is_finite(self, kappa):
        gs.random.seed(0)
        point = gs.to_numpy(
            self.space.random_von_mises_fisher(n_samples=100, kappa=kappa)
        )

        assert point.shape == (100, 2)
        assert np.isfinite(point).all()
        assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-10)
        assert np.all(point[:, 0] > 0.0)

    @pytest.mark.smoke
    @pytest.mark.shape
    def test_other_dimensions_are_unchanged(self):
        for dim in [2, 3, 4]:
            space = Hypersphere(dim)
            point = gs.to_numpy(space.random_von_mises_fisher(n_samples=50, kappa=5.0))
            assert point.shape == (50, dim + 1)
            assert np.allclose(np.linalg.norm(point, axis=-1), 1.0, atol=1e-6)
