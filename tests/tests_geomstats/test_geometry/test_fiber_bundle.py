import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.fiber_bundle import DistanceMinimizingAligner
from geomstats.geometry.general_linear import PositiveGeneralLinear
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.test_case import autodiff_backend
from geomstats.test_cases.geometry.fiber_bundle import (
    AlignComparisonTestCase,
    FiberBundleTestCase,
)
from geomstats.variants.geometry.spd_matrices import (  # noqa: F401
    PositiveGeneralLinearGramBundle,
)

from .data.fiber_bundle import AlignComparisonTestData, FiberBundleTestData


class TestPositiveGeneralLinearGramBundle(
    FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)

    total_space = PositiveGeneralLinear(_n, equip=False)

    total_space.equip_with_metric(MatricesMetric)
    total_space.equip_with_group_action(SpecialOrthogonalComposeAction(total_space.n))
    total_space.equip_with_quotient()

    base_space = SPDMatrices(n=_n, equip=False)

    testing_data = FiberBundleTestData(
        total_space=total_space,
        base_space=base_space,
        excluded_methods=(
            "integrability_tensor",
            "integrability_tensor_derivative",
        ),
    ).with_redundancy(projection_source="submersion_lift")


@pytest.fixture(
    scope="class",
    params=[False, True] if autodiff_backend() else [False],
)
def distance_minimizing_aligns(request):
    request.cls.other_align = DistanceMinimizingAligner(
        request.cls.total_space, use_log_gradient=request.param
    ).align


@pytest.mark.usefixtures("distance_minimizing_aligns")
class TestPositiveGeneralLinearGramBundleAlignCmp(
    AlignComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 4)

    total_space = PositiveGeneralLinear(_n, equip=False)

    total_space.equip_with_metric(MatricesMetric)
    total_space.equip_with_group_action(SpecialOrthogonalComposeAction(total_space.n))
    total_space.equip_with_quotient()

    align = total_space.fiber_bundle.align

    testing_data = AlignComparisonTestData(
        total_space, method_defaults={"align": {"atol": 1e-3}}
    )
