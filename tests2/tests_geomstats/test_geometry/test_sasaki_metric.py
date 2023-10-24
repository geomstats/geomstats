import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.sasaki_metric import SasakiMetric, TangentBundle
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.sasaki_metric import SasakiMetricTestCase

from .data.sasaki_metric import SasakiMetricSphereTestData


@pytest.mark.smoke
class TestSasakiMetricSphere(SasakiMetricTestCase, metaclass=DataBasedParametrizer):
    space = TangentBundle(Hypersphere(dim=2), equip=False)
    space.equip_with_metric(SasakiMetric)
    testing_data = SasakiMetricSphereTestData()
