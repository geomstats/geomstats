import random

import pytest

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


class TestSphericalToExtrinsicDiffeo(
    DiffeoTestCase, metaclass=DataBasedParametrizer
):
    space = SphereSphericalCoordinates(equip=False)
    image_space = Hypersphere(2, equip=False)
    diffeo = SphericalToExtrinsicDiffeo()
    testing_data = DiffeoTestData()


class TestHypersphere(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Hypersphere(random.randint(1, 5), equip=False)
    testing_data = LevelSetTestData()


@pytest.mark.smoke
class TestHypersphere2Metric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(2, equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere2MetricTestData()

    def test_log_antipodal_raises(self):
        """Log of antipodal points is not well-defined."""
        x = gs.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="antipodal"):
            self.space.metric.log(-x, x)


@pytest.mark.smoke
class TestHypersphere4Metric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(4, equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere4MetricTestData()

    def test_exp_and_dist_and_projection_to_tangent_space(self, vector, base_point):
        tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)
        exp = self.space.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = self.space.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)


class TestHypersphereMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
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
        2,
        random.randint(3, 5),
    ],
)
def von_mises_fisher_kappa_distributions(request):
    dim = request.param
    request.cls.distribution = VonMisesFisher(Hypersphere(dim, equip=False), kappa=1.0)


@pytest.mark.usefixtures("von_mises_fisher_kappa_distributions")
class TestVonMisesFisherKappa(
    VonMisesFisherTestCase, metaclass=DataBasedParametrizer
):
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
