import random

from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.spd_matrices import SPDBuresWassersteinMetric, SPDMatrices
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)
from geomstats.variants.geometry.spd_matrices import QuotientSPDBuresWassersteinMetric

from .data.riemannian_metric import RiemannianMetricComparisonTestData


class TestSPDBuresWassersteinQuotientMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)

    space = SPDMatrices(n=_n, equip=False).equip_with_metric(SPDBuresWassersteinMetric)
    other_space = SPDMatrices(n=_n, equip=False).equip_with_metric(
        QuotientSPDBuresWassersteinMetric
    )

    testing_data = RiemannianMetricComparisonTestData(
        space,
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
        ),
    )
