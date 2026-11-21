import numpy as np

import geomstats.backend as gs
from geomstats.test.test_case import torch_only
from geomstats.test_cases.backend.backend import (
    BackendDataBasedParametrizer,
    BackendTestCase,
)

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


@torch_only
def test_to_numpy_with_grad():
    """Check that to_numpy converts a tensor that requires gradients."""
    x = gs.ones(3).requires_grad_()

    assert isinstance(gs.to_numpy(2 * x), np.ndarray)


class TestBackend(BackendTestCase, metaclass=BackendDataBasedParametrizer):
    testing_data = BackendTestData()
