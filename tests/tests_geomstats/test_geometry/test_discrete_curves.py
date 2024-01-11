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
    SRVRotationReparametrizationBundle,
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
    SRVReparametrizationBundleTestCase,
)
from geomstats.test_cases.geometry.nfold_manifold import (
    NFoldManifoldTestCase,
    NFoldMetricTestCase,
)
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase

from .data.diffeo import (
    AutodiffDiffeoTestData,
    DiffeoComparisonTestData,
    DiffeoTestData,
)
from .data.discrete_curves import (
    AlignerCmpTestData,
    DiscreteCurvesStartingAtOriginTestData,
    ElasticMetricTestData,
    L2CurvesMetricTestData,
    SRVMetricTestData,
    SRVReparametrizationBundleTestData,
    SRVRotationBundleTestData,
    SRVRotationReparametrizationBundleTestData,
)


class TestDiscreteCurvesStartingAtOrigin(
    NFoldManifoldTestCase, metaclass=DataBasedParametrizer
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


class TestElasticMetric(PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer):
    _k_sampling_points = random.randint(5, 10)
    _a = gs.random.uniform(low=0.5, high=2.5, size=1)

    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=_k_sampling_points, equip=False
    ).equip_with_metric(ElasticMetric, a=_a)

    testing_data = ElasticMetricTestData()


class TestSRVMetric(PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim,
        k_sampling_points=_k_sampling_points,
    )
    testing_data = SRVMetricTestData()


class TestSRVReparametrizationBundle(
    SRVReparametrizationBundleTestCase, metaclass=DataBasedParametrizer
):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    total_space = base = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim,
        k_sampling_points=_k_sampling_points,
    )
    bundle = SRVReparametrizationBundle(total_space)

    data_generator = base_data_generator = ShapeBundleRandomDataGenerator(total_space)
    testing_data = SRVReparametrizationBundleTestData()

    def test_align(self, n_points, atol):
        base_point = self.total_space.random_point(n_points)
        base_curve = self.total_space.interpolate(base_point)
        k_sampling_points = self.total_space.k_sampling_points
        sampling_points = gs.linspace(0.0, 1.0, k_sampling_points)

        if n_points == 1:
            point = base_curve(sampling_points**2)
        else:
            point = gs.stack(
                [_base_curve(sampling_points**2) for _base_curve in base_curve]
            )

        point = self.total_space.projection(point)
        aligned_point = self.bundle.align(point, base_point)

        self.assertAllClose(aligned_point, base_point, atol=atol)


@pytest.mark.smoke
class TestAlignerCmp(TestCase, metaclass=DataBasedParametrizer):
    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=10)
    bundle = SRVReparametrizationBundle(total_space)

    aligner = IterativeHorizontalGeodesicAligner()
    other_aligner = DynamicProgrammingAligner()

    testing_data = AlignerCmpTestData()

    def test_align(self, curve_a, curve_b, atol):
        k_sampling_points = self.total_space.k_sampling_points
        sampling_points = gs.linspace(0.0, 1.0, k_sampling_points)

        base_point = self.total_space.projection(curve_a(sampling_points))
        point = self.total_space.projection(curve_b(sampling_points))

        aligned = self.aligner.align(self.bundle, point, base_point)
        other_aligned = self.other_aligner.align(self.bundle, point, base_point)

        self.assertAllClose(aligned, other_aligned, atol=atol)


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


class TestSRVRotationReparametrizationBundle(TestCase, metaclass=DataBasedParametrizer):
    _ambient_dim = random.randint(2, 3)
    _k_sampling_points = random.randint(5, 10)

    total_space = base = DiscreteCurvesStartingAtOrigin(
        ambient_dim=_ambient_dim,
        k_sampling_points=_k_sampling_points,
    )
    bundle = SRVRotationReparametrizationBundle(total_space)

    testing_data = SRVRotationReparametrizationBundleTestData()

    def test_align(self, n_points, atol):
        base_point = self.total_space.random_point(n_points)

        base_curve = self.total_space.interpolate(base_point)
        k_sampling_points = self.total_space.k_sampling_points
        sampling_points = gs.linspace(0.0, 1.0, k_sampling_points)

        if n_points == 1:
            point = base_curve(sampling_points**2)
        else:
            point = gs.stack(
                [_base_curve(sampling_points**2) for _base_curve in base_curve]
            )

        point = self.total_space.projection(point)
        rotation = SpecialOrthogonal(self._ambient_dim).random_point(n_points)
        point = self.bundle._rotations_bundle._rotate(point, rotation)

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
