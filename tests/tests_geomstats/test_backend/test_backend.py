import math

import pytest

import geomstats.backend as gs
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import BackendTestCase

from .data.backend import BackendTestData


def test_where_without_values():
    """Check that where without values returns matching indices."""
    (indices,) = gs.where([False, True, False, True])

    assert gs.all(indices == gs.array([1, 3]))


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (2, 1), (1, 2), (2, 3)])
def test_vec_to_diag_keeps_batch_shape(batch_shape, dim):
    """Check that vec_to_diag does not drop axes of size one."""
    size = math.prod(batch_shape) * dim
    vec = gs.reshape(gs.arange(1.0, size + 1.0), batch_shape + (dim,))

    diag_mat = gs.vec_to_diag(vec)
    assert diag_mat.shape == batch_shape + (dim, dim)

    flat_diag_mat = gs.reshape(diag_mat, (-1, dim, dim))
    for single_vec, single_diag_mat in zip(gs.reshape(vec, (-1, dim)), flat_diag_mat):
        assert gs.allclose(single_diag_mat, gs.eye(dim) * single_vec)


class TestBackend(BackendTestCase, metaclass=DataBasedParametrizer):
    testing_data = BackendTestData()
    
