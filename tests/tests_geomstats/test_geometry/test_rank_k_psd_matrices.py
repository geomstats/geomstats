import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.rank_k_psd_matrices import (
    BuresWassersteinBundle,
    PSDBuresWassersteinMetric,
    PSDMatrices,
    RankKPSDMatrices,
)
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.fiber_bundle import FiberBundleTestData
from .data.rank_k_psd_matrices import (
    Rank2PSD2BuresWassersteinMetricTestData,
    Rank2PSD3MatricesTestData,
    Rank3PSD3BuresWassersteinMetricTestData,
    RankKPSDMatricesTestData,
)
from .data.riemannian_metric import RiemannianMetricTestData


def _get_random_params():
    n = random.randint(3, 6)
    k = random.randint(2, n - 1)
    return n, k


@pytest.mark.smoke
class TestRank2PSD3Matrices(ManifoldTestCase, metaclass=DataBasedParametrizer):
    space = RankKPSDMatrices(n=3, k=2, equip=False)
    testing_data = Rank2PSD3MatricesTestData()


class TestRankKPSDMatrices(
    ProjectionTestCaseMixins,
    ManifoldTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n, _k = _get_random_params()
    space = RankKPSDMatrices(n=_n, k=_k, equip=False)
    testing_data = RankKPSDMatricesTestData(space)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
        _get_random_params(),
    ],
)
def bundle_spaces(request):
    if isinstance(request.param, int):
        n = k = request.param
    else:
        n, k = request.param

    base_space = request.cls.base_space = PSDMatrices(n=n, k=k, equip=False)

    total_space = request.cls.total_space = FullRankMatrices(n=n, k=k, equip=False)
    total_space.equip_with_metric(MatricesMetric)
    total_space.equip_with_group_action(SpecialOrthogonalComposeAction(total_space.k))
    total_space.equip_with_quotient()

    request.cls.testing_data.total_space = total_space
    request.cls.testing_data.base_space = base_space


@pytest.mark.usefixtures("bundle_spaces")
class TestBuresWassersteinBundle(
    FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = FiberBundleTestData(
        excluded_methods=(
            "integrability_tensor",
            "integrability_tensor_derivative",
        ),
    ).with_redundancy(projection_source="vertical")


@pytest.mark.smoke
class TestRank2PSD2BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PSDMatrices(n=2, k=2, equip=False)
    space.equip_with_metric(PSDBuresWassersteinMetric)

    testing_data = Rank2PSD2BuresWassersteinMetricTestData()


@pytest.mark.smoke
class TestRank3PSD3BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PSDMatrices(n=3, k=3, equip=False)
    space.equip_with_metric(PSDBuresWassersteinMetric)

    testing_data = Rank3PSD3BuresWassersteinMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
        _get_random_params(),
    ],
)
def spaces_with_quotient_metric(request):
    if isinstance(request.param, int):
        n = k = request.param
    else:
        n, k = request.param
    space = request.cls.space = PSDMatrices(n=n, k=k, equip=False)
    space.equip_with_metric(PSDBuresWassersteinMetric)

    request.cls.testing_data.space = space


@pytest.mark.redundant
@pytest.mark.usefixtures("spaces_with_quotient_metric")
class TestPSDBuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricTestData(
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
        generator_kwargs={"amplitude": 4.0},
    )


@pytest.mark.smoke
@pytest.mark.parametrize("n,k", [(3, 3), (3, 2)])
def test_psd_bures_wasserstein_metric_structure(n, k):
    """Check the default total space of the metric."""
    space = PSDMatrices(n=n, k=k, equip=False).equip_with_metric(
        PSDBuresWassersteinMetric
    )
    total_space = space.metric._total_space

    assert isinstance(total_space, FullRankMatrices)
    assert (total_space.n, total_space.k) == (n, k)
    assert isinstance(total_space.metric, MatricesMetric)
    assert isinstance(total_space.group_action, SpecialOrthogonalComposeAction)
    assert isinstance(total_space.fiber_bundle, BuresWassersteinBundle)
