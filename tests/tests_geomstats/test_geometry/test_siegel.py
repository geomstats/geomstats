import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.siegel import Siegel, SiegelMetric
from geomstats.test_cases.geometry.base import ComplexVectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.siegel import SiegelMetricTestCase

from .data.base import ComplexVectorSpaceOpenSetTestData
from .data.siegel import (
    Siegel2MetricTestData,
    Siegel2TestData,
    Siegel3MetricTestData,
    SiegelMetricTestData,
)


@pytest.mark.smoke
class TestSiegel2(ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=False)
    testing_data = Siegel2TestData()


class TestSiegel(
    ComplexVectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Siegel(random.randint(2, 5), equip=False)
    testing_data = ComplexVectorSpaceOpenSetTestData(space)


@pytest.mark.smoke
class TestSiegel2Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=False)
    space.equip_with_metric(SiegelMetric)
    testing_data = Siegel2MetricTestData()


@pytest.mark.smoke
class TestSiegel3Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(3, equip=False)
    space.equip_with_metric(SiegelMetric)
    testing_data = Siegel3MetricTestData()


class TestSiegelMetric(SiegelMetricTestCase, metaclass=GeometricDataBasedParametrizer):
    space = Siegel(random.randint(2, 5))
    testing_data = SiegelMetricTestData(
        space,
        excluded_methods=(
            "inner_product_is_complex",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
    )
