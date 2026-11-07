import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    ElasticMetric,
    FTransform,
    IterativeHorizontalGeodesicAligner,
    L2CurvesMetric,
    ReparametrizationBundle,
    SRVTransform,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.landmarks import Landmarks
from geomstats.test_cases.geometry.diffeo import (
    DiffeoComparisonTestCase,
    DiffeoTestCase,
)
from geomstats.test_cases.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOriginTestCase,
    ElasticMetricTestCase,
)
from geomstats.test_cases.geometry.fiber_bundle import (
    AlignComparisonTestCase,
    FiberBundleTestCase,
)
from geomstats.test_cases.geometry.nfold_manifold import NFoldMetricTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.variants.geometry.discrete_curves import (
    DynamicProgrammingAligner as NonVectorizedDynamicProgrammingAligner,
)

from .data.diffeo import DiffeoComparisonTestData, DiffeoTestData
from .data.discrete_curves import (
    DiscreteCurvesStartingAtOriginTestData,
    ElasticMetricDistanceTestData,
)
from .data.fiber_bundle import (
    AlignComparisonTestData,
    AlignInSameFiberTestData,
    FiberBundleTestData,
)
from .data.quotient_metric import AlignQuotientMetricTestData
from .data.riemannian_metric import RiemannianMetricTestData


def _curves_bundle(group_action):
    total_space = DiscreteCurvesStartingAtOrigin(
        random.randint(2, 3), random.randint(5, 10)
    )
    total_space.equip_with_group_action(group_action)
    total_space.equip_with_quotient()
    return total_space


def _curves_and_landmarks(ambient_dim, k_sampling_points):
    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=ambient_dim, k_sampling_points=k_sampling_points, equip=False
    )
    image_space = Landmarks(
        ambient_manifold=space.ambient_manifold,
        k_landmarks=space.k_sampling_points - 1,
        equip=False,
    )
    return space, image_space


class TestDiscreteCurvesStartingAtOrigin(
    DiscreteCurvesStartingAtOriginTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=random.randint(2, 3),
        k_sampling_points=random.randint(5, 10),
        equip=False,
    )
    testing_data = DiscreteCurvesStartingAtOriginTestData(
        space, excluded_methods=("not_belongs",)
    )


class TestSRVTransform(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _k_sampling_points = random.randint(5, 10)
    domain_space, image_space = _curves_and_landmarks(
        random.randint(2, 3), _k_sampling_points
    )
    diffeo = SRVTransform(domain_space.ambient_manifold, _k_sampling_points)

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestFTransform(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _k_sampling_points = random.randint(5, 10)
    domain_space, image_space = _curves_and_landmarks(2, _k_sampling_points)
    diffeo = FTransform(
        domain_space.ambient_manifold,
        _k_sampling_points,
        gs.random.uniform(low=0.5, high=2.5, size=1),
    )

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestFTransformAsSRVCmp(
    DiffeoComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _k_sampling_points = random.randint(5, 10)
    domain_space, image_space = _curves_and_landmarks(2, _k_sampling_points)
    diffeo = SRVTransform(domain_space.ambient_manifold, _k_sampling_points)
    other_diffeo = FTransform(domain_space.ambient_manifold, _k_sampling_points)

    testing_data = DiffeoComparisonTestData(
        domain_space=domain_space,
        image_space=image_space,
    )


class TestL2CurvesMetric(NFoldMetricTestCase, metaclass=GeometricDataBasedParametrizer):
    space = Landmarks(
        ambient_manifold=Euclidean(dim=random.randint(2, 3)),
        k_landmarks=random.randint(5, 10),
        equip=False,
    ).equip_with_metric(L2CurvesMetric)

    testing_data = RiemannianMetricTestData(space)


@pytest.fixture(
    scope="class",
    params=[
        (2, random.randint(5, 10), random.uniform(0.5, 2.5)),
        (random.randint(3, 5), random.randint(5, 10), random.uniform(0.5, 2.5)),
    ],
)
def discrete_curves_with_elastic(request):
    """Equip discrete curves with elastic metrics with :math:`a = 2b`.

    Planar curves use the F transform, other curves a scaled SRV metric.
    The F transform is only a diffeomorphism onto the whole image space when
    :math:`a / (2b) = 1`.
    """
    ambient_dim, k_sampling_points, b = request.param
    a = 2 * b

    space = request.cls.space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=ambient_dim, k_sampling_points=k_sampling_points, equip=False
    ).equip_with_metric(ElasticMetric, a=a, b=b)

    request.cls.testing_data.space = space


@pytest.mark.redundant
@pytest.mark.usefixtures("discrete_curves_with_elastic")
class TestElasticMetric(
    PullbackDiffeoMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricTestData()


@pytest.mark.usefixtures("discrete_curves_with_elastic")
class TestElasticMetricDistance(
    ElasticMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = ElasticMetricDistanceTestData()


@pytest.mark.redundant
class TestSRVMetric(
    PullbackDiffeoMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = DiscreteCurvesStartingAtOrigin(
        ambient_dim=random.randint(2, 3),
        k_sampling_points=random.randint(5, 10),
    )
    testing_data = RiemannianMetricTestData(space)


@pytest.fixture(
    scope="class",
    params=[random.randrange(40, 62, 2), random.randrange(41, 62, 2)],
)
def horizontal_aligners(request):
    total_space = request.cls.total_space = DiscreteCurvesStartingAtOrigin(
        k_sampling_points=request.param
    )
    total_space.fiber_bundle = ReparametrizationBundle(
        total_space,
        aligner=IterativeHorizontalGeodesicAligner(total_space),
    )

    request.cls.testing_data.space = total_space


@pytest.mark.usefixtures("horizontal_aligners")
class TestIterativeHorizontalGeodesicAligner(
    FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = AlignInSameFiberTestData(
        point_counts=[1],
        generator_kwargs={"fiber_actions": ("reparametrizations",)},
        method_defaults={"align_in_same_fiber": {"atol": 1e-5}},
    )


class TestDynamicProgrammingAligner(
    FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    total_space = DiscreteCurvesStartingAtOrigin(
        k_sampling_points=random.randint(40, 60)
    )
    total_space.fiber_bundle = ReparametrizationBundle(
        total_space, aligner=DynamicProgrammingAligner(total_space)
    )

    testing_data = AlignInSameFiberTestData(
        total_space,
        point_counts=[1],
        generator_kwargs={"fiber_actions": ("reparametrizations",)},
        method_defaults={"align_in_same_fiber": {"atol": 1e-1}},
    )


@pytest.mark.redundant
@pytest.mark.xfail
class TestReparametrizationAlignerCmp(
    AlignComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    total_space = DiscreteCurvesStartingAtOrigin(
        k_sampling_points=random.randint(40, 50)
    )
    total_space.fiber_bundle = ReparametrizationBundle(total_space)

    align = IterativeHorizontalGeodesicAligner(total_space).align
    other_align = DynamicProgrammingAligner(total_space).align

    testing_data = AlignComparisonTestData(
        total_space,
        point_counts=[1],
        method_defaults={"align": {"atol": 1e-3}},
    )


@pytest.mark.redundant
class TestDynamicProgrammingAlignerVariantCmp(
    AlignComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=random.randint(4, 8))

    align = DynamicProgrammingAligner(total_space).align
    other_align = NonVectorizedDynamicProgrammingAligner(total_space).align

    testing_data = AlignComparisonTestData(
        total_space,
        point_counts=[1],
        method_defaults={"align": {"atol": 1e-6}},
    )


# NB: `DataBasedParametrizer` skips vectorization tests, which are slow here
class TestReparametrizationBundle(FiberBundleTestCase, metaclass=DataBasedParametrizer):
    total_space = base_space = DiscreteCurvesStartingAtOrigin(
        random.randint(2, 3), random.randint(20, 30)
    )
    total_space.fiber_bundle = ReparametrizationBundle(
        total_space,
        IterativeHorizontalGeodesicAligner(total_space),
    )

    testing_data = FiberBundleTestData(
        total_space=total_space,
        base_space=base_space,
        method_marks={
            "xfail": (
                "tangent_riemannian_submersion_after_horizontal_lift",
                "log_after_align_is_horizontal",
                "projections_are_orthogonal",
            )
        },
        method_defaults={
            "projections_are_orthogonal": {"atol": 1e-2},
            "log_after_align_is_horizontal": {"atol": 1e-2},
            "vertical_projection_is_vertical": {"atol": 1e-1},
            "horizontal_projection_is_horizontal": {"atol": 1e-1},
            "horizontal_lift_is_horizontal": {"atol": 1e-1},
            "tangent_riemannian_submersion_after_vertical_projection": {"atol": 5e-1},
            "tangent_riemannian_submersion_after_horizontal_lift": {"atol": 5e-1},
        },
        point_counts=[1, 2],
    ).with_redundancy(
        projection_source="vertical", lift_uses_horizontal_projection=True
    )


class TestRotationAlign(FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer):
    total_space = _curves_bundle("rotations")

    testing_data = AlignInSameFiberTestData(
        total_space, generator_kwargs={"fiber_actions": ("rotations",)}
    )


@pytest.mark.xfail
class TestRotationReparametrizationAlign(
    FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    total_space = _curves_bundle(("rotations", "reparametrizations"))
    testing_data = AlignInSameFiberTestData(
        total_space,
        generator_kwargs={"fiber_actions": ("rotations", "reparametrizations")},
        method_defaults={"align_in_same_fiber": {"atol": 1e-2}},
    )


def _curves_quotient(group_action):
    return _curves_bundle(group_action).quotient


@pytest.mark.redundant
@pytest.mark.slow
@pytest.mark.xfail
class TestReparametrizationQuotientMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = _curves_quotient("reparametrizations")
    testing_data = RiemannianMetricTestData(
        space,
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
    )


@pytest.mark.redundant
class TestRotationQuotientMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = _curves_quotient("rotations")
    testing_data = AlignQuotientMetricTestData(
        space,
        method_marks={
            "xfail": ("geodesic_bvp_reverse", "geodesic_boundary_points"),
        },
    )


@pytest.mark.redundant
class TestRotationReparametrizationQuotientMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = _curves_quotient(("rotations", "reparametrizations"))
    testing_data = AlignQuotientMetricTestData(
        space,
        method_marks={
            "xfail": (
                "geodesic_bvp_reverse",
                "geodesic_boundary_points",
            )
        },
        method_defaults={
            "dist_is_symmetric": {"atol": 1e-2},
            "squared_dist_is_symmetric": {"atol": 1e-2},
        },
    )
