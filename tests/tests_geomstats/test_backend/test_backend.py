import geomstats.backend as gs
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import BackendTestCase

from .data.backend import BackendTestData


def test_where_without_values():
    """Check that where without values returns matching indices."""
    (indices,) = gs.where([False, True, False, True])

    assert gs.all(indices == gs.array([1, 3]))


class TestBackend(BackendTestCase, metaclass=DataBasedParametrizer):
    testing_data = BackendTestData()
