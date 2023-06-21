from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)

from .base import OpenSetTestData


class PositiveRealsTestData(OpenSetTestData):
    pass


class PositiveRealsMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
