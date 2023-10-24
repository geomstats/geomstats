import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ImmersedSetTestCase
from geomstats.test_cases.geometry.pullback_metric import (
    CircleAsSO2Metric,
    CircleIntrinsic,
    PullbackMetricTestCase,
    SphereIntrinsic,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)

from .data.pullback_metric import (
    CircleAsSO2PullbackDiffeoMetricCmpTestData,
    CircleIntrinsicMetricTestData,
    CircleIntrinsicTestData,
    SphereIntrinsicMetricTestData,
    SphereIntrinsicTestData,
)


@pytest.mark.smoke
class TestCircleIntrinsic(ImmersedSetTestCase, metaclass=DataBasedParametrizer):
    space = CircleIntrinsic(equip=False)
    testing_data = CircleIntrinsicTestData()


@pytest.mark.smoke
class TestSphereIntrinsic(ImmersedSetTestCase, metaclass=DataBasedParametrizer):
    space = SphereIntrinsic(equip=False)
    testing_data = SphereIntrinsicTestData()


@pytest.mark.smoke
class TestCircleIntrinsicMetric(
    PullbackMetricTestCase, metaclass=DataBasedParametrizer
):
    space = CircleIntrinsic()
    testing_data = CircleIntrinsicMetricTestData()


@pytest.mark.smoke
class TestSphereIntrinsicMetric(
    PullbackMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SphereIntrinsic()
    testing_data = SphereIntrinsicMetricTestData()

    def test_second_fundamental_form(self, base_point, expected_11, expected_22, atol):
        res = self.space.metric.second_fundamental_form(base_point)

        res_11 = res[:, 0, 0]
        res_22 = res[:, 1, 1]

        self.assertAllClose(res_11, expected_11, atol=atol)
        self.assertAllClose(res_22, expected_22, atol=atol)

    def test_inner_product_derivative_matrix(
        self, base_point, expected_1, expected_2, atol
    ):
        res = self.space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res[:, :, 0], expected_1, atol=atol)
        self.assertAllClose(res[:, :, 1], expected_2, atol=atol)


class TestCircleAsSO2PullbackDiffeoMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    space = Hypersphere(dim=1)

    other_space = Hypersphere(dim=1, equip=False)
    other_space.equip_with_metric(CircleAsSO2Metric)

    testing_data = CircleAsSO2PullbackDiffeoMetricCmpTestData()
