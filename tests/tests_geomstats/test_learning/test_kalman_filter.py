import pytest

from geomstats.learning.kalman_filter import (
    KalmanFilter,
    Localization,
    LocalizationLinear,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.kalman_filter import (
    KalmanFilterTestCase,
    LocalizationTestCase,
    NonLinearLocalizationTestCase,
)

from .data.kalman_filter import (
    KalmanFilterTestData,
    LocalizationLinearTestData,
    LocalizationTestData,
)


@pytest.mark.smoke
class TestLocalizationLinear(
    LocalizationTestCase,
    metaclass=DataBasedParametrizer,
):
    model = LocalizationLinear()
    testing_data = LocalizationLinearTestData()


@pytest.mark.smoke
class TestLocalization(
    NonLinearLocalizationTestCase,
    metaclass=DataBasedParametrizer,
):
    model = Localization()
    testing_data = LocalizationTestData()


@pytest.mark.smoke
class TestKalmanFilter(
    KalmanFilterTestCase,
    metaclass=DataBasedParametrizer,
):
    estimator = KalmanFilter(LocalizationLinear())
    testing_data = KalmanFilterTestData()
