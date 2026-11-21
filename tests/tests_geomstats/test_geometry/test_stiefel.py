import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from geomstats.test_cases.geometry.retraction import RetractionTestCase
from geomstats.test_cases.geometry.stiefel import (
    StiefelCanonicalMetricTestCase,
    StiefelStaticMethodsTestCase,
    StiefelTestCase,
)

from .data.retraction import RetractionTestData
from .data.stiefel import (
    SquareStiefelCanonicalMetricTestData,
    StiefelCanonicalMetric42TestData,
    StiefelCanonicalMetricTestData,
    StiefelStaticMethodsTestData,
    StiefelTestData,
)

# TODO: use comparison with limit cases
# TODO: make it work with p=1?


def _get_random_params(excluded=()):
    while True:
        a = random.randint(2, 5)
        b = random.randint(2, 5)
        if a == b:
            continue

        n, p = max(a, b), min(a, b)
        if (n, p) not in excluded:
            return n, p


class TestStiefel(StiefelTestCase, metaclass=GeometricDataBasedParametrizer):
    _n, _p = _get_random_params()
    space = Stiefel(n=_n, p=_p, equip=False)

    testing_data = StiefelTestData(space)


@pytest.mark.smoke
class TestStiefelStaticMethods(
    StiefelStaticMethodsTestCase, metaclass=DataBasedParametrizer
):
    Space = Stiefel
    testing_data = StiefelStaticMethodsTestData()


class TestStiefelCanonicalMetric(
    StiefelCanonicalMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n, _p = _get_random_params()
    space = Stiefel(n=_n, p=_p)

    testing_data = StiefelCanonicalMetricTestData(
        space,
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
        sample_locally=True,
    )

    def test_log_antipodal_raises(self, point):
        """Log of antipodal points is not well-defined."""
        with pytest.raises(ValueError, match="antipodal"):
            self.space.metric.log(-point, point)


@pytest.mark.smoke
class TestStiefelCanonicalMetric42(
    StiefelCanonicalMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Stiefel(4, 2)
    testing_data = StiefelCanonicalMetric42TestData()

    def test_log_partially_antipodal_raises(self, point, base_point):
        """Log is not well-defined if a column of point is opposite to base point's.

        The orthogonal completion then has eigenvalues equal to -1, so its
        principal logarithm is complex.
        """
        with pytest.raises(ValueError, match="imaginary"):
            self.space.metric.log(point, base_point)


class TestSquareStiefelCanonicalMetric(
    StiefelCanonicalMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    k = random.randint(2, 5)

    space = Stiefel(n=k, p=k, equip=False)
    space.equip_with_metric(StiefelCanonicalMetric)

    testing_data = SquareStiefelCanonicalMetricTestData(
        space,
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
    )

    def test_two_sheets_error(self, point, base_point):
        """Log of points on different sheets raises.

        The first row of `point` is flipped, if needed, so that `point`
        and `base_point` lie on different connected components.
        """
        det_point = gs.linalg.det(point)
        det_base = gs.linalg.det(base_point)

        sign = gs.where(det_base * det_point > 0.0, -1.0, 1.0)
        first_row = sign[..., None, None] * point[..., :1, :]
        point = gs.concatenate([first_row, point[..., 1:, :]], axis=-2)

        with pytest.raises(ValueError):
            self.space.metric.log(point, base_point)


class TestStiefelRetraction(
    RetractionTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n, _p = _get_random_params()
    space = Stiefel(n=_n, p=_p)

    testing_data = RetractionTestData(space)
