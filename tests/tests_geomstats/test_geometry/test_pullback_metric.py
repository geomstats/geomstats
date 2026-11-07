import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.invariant_metric import BiInvariantMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.test_case import autodiff_backend, autodiff_only
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
    CircleIntrinsicMetricTestData,
    CircleIntrinsicTestData,
    SphereIntrinsicMetricTestData,
    SphereIntrinsicTestData,
)
from .data.riemannian_metric import RiemannianMetricComparisonTestData


@autodiff_only
@pytest.mark.smoke
class TestCircleIntrinsic(ImmersedSetTestCase, metaclass=DataBasedParametrizer):
    space = CircleIntrinsic(equip=False)
    testing_data = CircleIntrinsicTestData()


@pytest.mark.smoke
class TestSphereIntrinsic(ImmersedSetTestCase, metaclass=DataBasedParametrizer):
    space = SphereIntrinsic(equip=False)
    testing_data = SphereIntrinsicTestData(
        excluded_methods=()
        if autodiff_backend()
        else ("tangent_immersion", "jacobian_immersion", "hessian_immersion"),
    )


@autodiff_only
@pytest.mark.smoke
class TestCircleIntrinsicMetric(
    PullbackMetricTestCase, metaclass=DataBasedParametrizer
):
    space = CircleIntrinsic()
    testing_data = CircleIntrinsicMetricTestData()


@autodiff_only
@pytest.mark.smoke
class TestSphereIntrinsicMetric(
    PullbackMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SphereIntrinsic()
    testing_data = SphereIntrinsicMetricTestData()

    def test_second_fundamental_form(
        self, base_point, expected_11, expected_22, atol=gs.atol
    ):
        """Check the diagonal of the second fundamental form.

        Parameters
        ----------
        base_point : array-like, shape=[2]
            Point on the sphere, in spherical coordinates.
        expected_11 : array-like, shape=[3]
            Expected first diagonal component.
        expected_22 : array-like, shape=[3]
            Expected second diagonal component.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.second_fundamental_form(base_point)

        res_11 = res[:, 0, 0]
        res_22 = res[:, 1, 1]

        self.assertAllClose(res_11, expected_11, atol=atol)
        self.assertAllClose(res_22, expected_22, atol=atol)

    def test_inner_product_derivative_matrix(
        self, base_point, expected_1, expected_2, atol=gs.atol
    ):
        """Check the derivatives of the metric matrix.

        Parameters
        ----------
        base_point : array-like, shape=[2]
            Point on the sphere, in spherical coordinates.
        expected_1 : array-like, shape=[2, 2]
            Expected derivative with respect to the first coordinate.
        expected_2 : array-like, shape=[2, 2]
            Expected derivative with respect to the second coordinate.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res[:, :, 0], expected_1, atol=atol)
        self.assertAllClose(res[:, :, 1], expected_2, atol=atol)


@autodiff_only
class TestCircleAsSO2PullbackDiffeoMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Hypersphere(dim=1)

    _image_space = SpecialOrthogonal(n=2, point_type="matrix", equip=False)
    _image_space.equip_with_metric(BiInvariantMetric)

    other_space = Hypersphere(dim=1, equip=False)
    other_space.equip_with_metric(CircleAsSO2Metric, image_space=_image_space)

    testing_data = RiemannianMetricComparisonTestData(space)
