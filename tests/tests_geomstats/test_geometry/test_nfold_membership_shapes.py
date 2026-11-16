"""Regression tests for n-fold product membership shapes."""

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.nfold_manifold import NFoldManifold


@pytest.mark.parametrize(
    "shape",
    [(), (2,), (6,), (1, 6), (2, 3), (3, 3), (4, 2), (2, 1, 1, 6), (2, 1, 3, 3)],
)
def test_nfold_membership_rejects_wrong_point_shapes(shape):
    """Reject shape aliases without raising reshape errors."""
    space = NFoldManifold(Euclidean(2), n_copies=3)
    point = gs.ones(shape)

    result = space.belongs(point)

    assert result.shape == shape[:-space.point_ndim]
    assert not gs.any(result)


@pytest.mark.parametrize(
    "shape", [(12,), (2, 6), (3, 4), (3, 1, 4), (2, 1, 3, 1, 4)]
)
def test_nfold_matrix_membership_rejects_reshaped_coordinates(shape):
    """Require the matrix dimensions as well as the number of copies."""
    space = NFoldManifold(Matrices(2, 2), n_copies=3)

    result = space.belongs(gs.ones(shape))

    assert result.shape == shape[:-space.point_ndim]
    assert not gs.any(result)


@pytest.mark.parametrize("n_copies", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 1)])
def test_nfold_membership_preserves_valid_batch_axes(n_copies, batch_shape):
    """Return one membership value per point without dropping singleton axes."""
    space = NFoldManifold(Euclidean(2), n_copies=n_copies)

    result = space.belongs(gs.ones(batch_shape + space.shape))

    assert result.shape == batch_shape
    assert gs.all(result)


def test_nested_nfold_membership_checks_the_complete_point_shape():
    """Keep the copy axes of nested products distinct."""
    base = NFoldManifold(Euclidean(2), n_copies=3)
    space = NFoldManifold(base, n_copies=2)

    assert not space.belongs(gs.ones((2, 6)))
    assert space.belongs(gs.ones(space.shape))


def test_nfold_membership_still_checks_every_base_point():
    """Shape validation must retain the base manifold's geometric constraints."""
    space = NFoldManifold(Hypersphere(2), n_copies=2)
    points = gs.array(
        [
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        ]
    )

    result = space.belongs(points)

    assert result.shape == (2, 1)
    assert gs.all(result == gs.array([[True], [False]]))


def test_nfold_membership_forwards_tolerance_to_base_manifold():
    """Keep membership tolerance meaningful after checking the point shape."""
    space = NFoldManifold(Hypersphere(2), n_copies=2)
    point = gs.array([[1.0001, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert space.belongs(point, atol=1e-3)
    assert not space.belongs(point, atol=1e-8)
