import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    ElasticMetric,
    FTransform,
    IterativeHorizontalGeodesicAligner,
    L2CurvesMetric,
    SRVReparametrizationBundle,
    SRVRotationBundle,
    SRVTransform,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.landmarks import Landmarks
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import ShapeBundleRandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.diffeo import (
    AutodiffDiffeoTestCase,
    DiffeoComparisonTestCase,
    DiffeoTestCase,
)
from geomstats.test_cases.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOriginTestCase,
    ElasticMetricTestCase,
    SRVReparametrizationBundleTestCase,
)
from geomstats.test_cases.geometry.nfold_manifold import NFoldMetricTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase

from .data.diffeo import (
    AutodiffDiffeoTestData,
    DiffeoComparisonTestData,
    DiffeoTestData,
)
from .data.discrete_curves import (
    DiscreteCurvesStartingAtOriginTestData,
    ElasticMetricTestData,
    L2CurvesMetricTestData,
    ReparameterizationAlignerTestData,
    SRVMetricTestData,
    SRVReparametrizationBundleTestData,
    SRVReparametrizationsQuotientMetricTestData,
    SRVRotationBundleTestData,
    SRVRotationReparametrizationsBundleTestData,
    SRVRotationsAndReparametrizationsQuotientMetricTestData,
    SRVRotationsQuotientMetricTestData,
)


class TestDiscreteCurvesStartingAtOrigin(
    DiscreteCurvesStartingAtOriginTestCase, metaclass=DataBasedParametrizer
):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim, k_sampling_points=_k_sampling_points, equip=False
    )
    testing_data = DiscreteCurvesStartingAtOriginTestData()


class TestSRVTransform(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim, k_sampling_points=_k_sampling_points, equip=False
    )

    image_space = Landmarks(
        ambient_manifold=space.ambient_manifold,
        k_landmarks=space.k_sampling_points - 1,
        equip=False,
    )

    diffeo = SRVTransform(space.ambient_manifold, _k_sampling_points)

    testing_data = DiffeoTestData()


class TestFTransform(AutodiffDiffeoTestCase, metaclass=DataBasedParametrizer):
    _k_sampling_points = random.randint(5, 10)
    _a = gs.random.uniform(low=0.5, high=2.5, size=1)

    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=_k_sampling_points, equip=False
    )

    image_space = Landmarks(
        ambient_manifold=space.ambient_manifold,
        k_landmarks=space.k_sampling_points - 1,
        equip=False,
    )

    diffeo = FTransform(space.ambient_manifold, _k_sampling_points, _a)

    testing_data = AutodiffDiffeoTestData()


class TestSRVVsFTransform(DiffeoComparisonTestCase, metaclass=DataBasedParametrizer):
    _k_sampling_points = random.randint(5, 10)

    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=_k_sampling_points, equip=False
    )

    image_space = Landmarks(
        ambient_manifold=space.ambient_manifold,
        k_landmarks=space.k_sampling_points - 1,
        equip=False,
    )
    diffeo = SRVTransform(space.ambient_manifold, _k_sampling_points)
    other_diffeo = FTransform(space.ambient_manifold, _k_sampling_points)
    testing_data = DiffeoComparisonTestData()


class TestL2CurvesMetric(NFoldMetricTestCase, metaclass=DataBasedParametrizer):
    _dim = random.randint(2, 3)
    _k_landmarks = random.randint(5, 10)

    space = Landmarks(
        ambient_manifold=Euclidean(dim=_dim),
        k_landmarks=_k_landmarks,
        equip=False,
    ).equip_with_metric(L2CurvesMetric)

    testing_data = L2CurvesMetricTestData()


class TestSRVMetric(PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim,
        k_sampling_points=_k_sampling_points,
    )
    testing_data = SRVMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        (
            2,
            random.randint(5, 10),
            float(gs.random.uniform(low=0.5, high=2.5, size=1)[0]),
        ),
        (
            random.randint(3, 5),
            random.randint(5, 10),
            0.5,
        ),
        (
            random.randint(3, 5),
            random.randint(5, 10),
            float(gs.random.uniform(low=0.5, high=2.5, size=1)[0]),
        ),
    ],
)
def discrete_curves_with_elastic(request):
    ambient_dim, k_sampling_points, b = request.param

    lambda_ = 1.0
    a = lambda_ * 2 * b

    request.cls.space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=ambient_dim, k_sampling_points=k_sampling_points, equip=False
    ).equip_with_metric(ElasticMetric, a=a, b=b)


@pytest.mark.usefixtures("discrete_curves_with_elastic")
class TestElasticMetric(ElasticMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = ElasticMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(2, 3), random.choice([6, 8, 10])),
        (random.randint(2, 3), random.choice([5, 7, 9])),
    ],
)
def srv_reparameterization_bundles(request):
    ambient_dim, k_sampling_points = request.param

    total_space = request.cls.total_space = request.cls.base = (
        DiscreteCurvesStartingAtOrigin(ambient_dim, k_sampling_points)
    )
    total_space.fiber_bundle = SRVReparametrizationBundle(total_space)

    request.cls.data_generator = request.cls.base_data_generator = (
        ShapeBundleRandomDataGenerator(total_space)
    )


@pytest.mark.usefixtures("srv_reparameterization_bundles")
class TestSRVReparametrizationBundle(
    SRVReparametrizationBundleTestCase, metaclass=DataBasedParametrizer
):
    ambient_dim = random.randint(2, 3)
    k_sampling_points = random.randint(4, 8)

    total_space = base = DiscreteCurvesStartingAtOrigin(ambient_dim, k_sampling_points)
    total_space.fiber_bundle = SRVReparametrizationBundle(total_space)

    data_generator = base_data_generator = ShapeBundleRandomDataGenerator(total_space)
    testing_data = SRVReparametrizationBundleTestData()


@pytest.fixture(
    scope="class",
    params=[
        (DynamicProgrammingAligner, random.randint(4, 8)),
        (IterativeHorizontalGeodesicAligner, random.choice([4, 6, 8])),
        (IterativeHorizontalGeodesicAligner, random.choice([5, 7, 9])),
    ],
)
def aligners(request):
    Aligner, k_sampling_points = request.param

    request.cls.total_space = total_space = DiscreteCurvesStartingAtOrigin(
        k_sampling_points=k_sampling_points
    )

    aligner = Aligner(total_space)
    total_space.fiber_bundle = SRVReparametrizationBundle(total_space, aligner=aligner)


@pytest.mark.usefixtures("aligners")
class TestReparameterizationAligner(TestCase, metaclass=DataBasedParametrizer):
    testing_data = ReparameterizationAlignerTestData()

    def test_align_in_same_fiber(self, n_points, atol):
        base_point = self.total_space.random_point(n_points)
        base_curve = self.total_space.interpolate(base_point)
        k_sampling_points = self.total_space.k_sampling_points
        sampling_points = gs.linspace(0.0, 1.0, k_sampling_points)

        point = base_curve(sampling_points**2)

        point = self.total_space.projection(point)
        aligned_point = self.total_space.fiber_bundle.align(point, base_point)

        self.assertAllClose(aligned_point, base_point, atol=atol)


class TestSRVRotationBundle(TestCase, metaclass=DataBasedParametrizer):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    total_space = base = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim,
        k_sampling_points=_k_sampling_points,
    )
    bundle = SRVRotationBundle(total_space)

    testing_data = SRVRotationBundleTestData()

    def test_align(self, n_points, atol):
        base_point = self.total_space.random_point(n_points)

        rotation = SpecialOrthogonal(self._ambient_dim).random_point(n_points)
        point = self.bundle._rotate(base_point, rotation)

        aligned_point, inv_rotation = self.bundle.align(
            point, base_point, return_rotation=True
        )
        result = gs.matmul(rotation, inv_rotation)
        if n_points == 1:
            expected = gs.eye(self._ambient_dim)
        else:
            expected = gs.stack([gs.eye(self._ambient_dim) for _ in range(n_points)])

        self.assertAllClose(result, expected, atol=atol)
        self.assertAllClose(aligned_point, base_point, atol=atol)


class TestSRVRotationReparametrizationsBundle(
    TestCase, metaclass=DataBasedParametrizer
):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    total_space = base = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim,
        k_sampling_points=_k_sampling_points,
    )
    total_space.equip_with_group_action(("rotations", "reparametrizations"))
    total_space.equip_with_quotient()

    testing_data = SRVRotationReparametrizationsBundleTestData()

    def test_align(self, n_points, atol):
        base_point = self.total_space.random_point(n_points)

        base_curve = self.total_space.interpolate(base_point)
        k_sampling_points = self.total_space.k_sampling_points
        sampling_points = gs.linspace(0.0, 1.0, k_sampling_points)

        point = base_curve(sampling_points**2)
        point = self.total_space.projection(point)

        rotation_bundle = self.total_space.fiber_bundle.total_spaces[0].fiber_bundle
        rotation = SpecialOrthogonal(self._ambient_dim).random_point(n_points)
        point = rotation_bundle._rotate(point, rotation)

        aligned_point = self.total_space.fiber_bundle.align(point, base_point)
        self.assertAllClose(aligned_point, base_point, atol=atol)


@pytest.mark.redundant
@pytest.mark.xfail
class TestSRVReparametrizationsQuotientMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)
    _total_space = DiscreteCurvesStartingAtOrigin(_ambient_dim, _k_sampling_points)
    _total_space.equip_with_group_action("reparametrizations")
    _total_space.equip_with_quotient()

    space = _total_space.quotient

    testing_data = SRVReparametrizationsQuotientMetricTestData()


@pytest.mark.redundant
class TestSRVRotationsQuotientMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)
    _total_space = DiscreteCurvesStartingAtOrigin(_ambient_dim, _k_sampling_points)
    _total_space.equip_with_group_action("rotations")
    _total_space.equip_with_quotient()

    space = _total_space.quotient

    testing_data = SRVRotationsQuotientMetricTestData()


@pytest.mark.redundant
class TestSRVRotationsReparametrizationsQuotientMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)
    _total_space = DiscreteCurvesStartingAtOrigin(_ambient_dim, _k_sampling_points)
    _total_space.equip_with_group_action(("rotations", "reparametrizations"))
    _total_space.equip_with_quotient()

    space = _total_space.quotient

    testing_data = SRVRotationsAndReparametrizationsQuotientMetricTestData()
