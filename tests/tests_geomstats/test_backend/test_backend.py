import geomstats.backend as gs
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import BackendTestCase

from .data.backend import BackendTestData


def test_where_without_values():
    """Check that where without values returns matching indices."""
    (indices,) = gs.where([False, True, False, True])

    assert gs.all(indices == gs.array([1, 3]))


def test_solve_sylvester_complex():
    """Check the solution of Sylvester equations with complex coefficients.

    The first coefficient is symmetric but not Hermitian, and the second is
    Hermitian positive definite.
    """
    q = gs.ones((2, 2), dtype=gs.get_default_cdtype())
    for a in (gs.array([[2j, 0.0], [0.0, 3j]]), gs.array([[2.0, 1j], [-1j, 2.0]])):
        x = gs.linalg.solve_sylvester(a, a, q)

        assert gs.allclose(gs.matmul(a, x) + gs.matmul(x, a), q)


class TestBackend(BackendTestCase, metaclass=DataBasedParametrizer):
    testing_data = BackendTestData()
