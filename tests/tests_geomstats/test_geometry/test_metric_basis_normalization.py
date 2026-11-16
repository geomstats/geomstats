"""Regression tests for metric normalization of vector and tensor bases."""

import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.nfold_manifold import NFoldManifold
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.spd_matrices import SPDMatrices


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1)])
@pytest.mark.parametrize("kind", ["vector", "matrix", "tensor"])
def test_normal_basis_preserves_arbitrary_point_and_batch_axes(kind, batch_shape):
    """Normalize each basis vector without interpreting point axes as batches."""
    space = {
        "vector": Euclidean(3),
        "matrix": Matrices(2, 3),
        "tensor": NFoldManifold(NFoldManifold(Euclidean(2), n_copies=2), n_copies=2),
    }[kind]
    basis = gs.reshape(gs.eye(space.dim), (space.dim,) + space.shape)
    weights = gs.reshape(
        gs.linspace(1.0, 2.0, space.dim),
        (space.dim,) + (1,) * space.point_ndim,
    )
    basis = gs.broadcast_to(weights * basis, batch_shape + basis.shape)
    original = gs.copy(basis)
    base_point = gs.zeros(space.shape)

    normalized = space.metric.normal_basis(basis, base_point)

    expected = gs.reshape(gs.eye(space.dim), (space.dim,) + space.shape)
    expected = gs.broadcast_to(expected, basis.shape)
    assert normalized.shape == basis.shape
    assert gs.allclose(normalized, expected)
    assert gs.allclose(space.metric.squared_norm(normalized, base_point), 1.0)
    assert gs.all(basis == original)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1)])
def test_normal_basis_uses_the_metric_at_each_base_point(batch_shape):
    """Normalize vector coordinates using the local Poincare conformal factor."""
    space = PoincareBall(3)
    base_point = gs.reshape(
        gs.linspace(0.01, 0.2, math.prod(batch_shape) * 3), batch_shape + (1, 3)
    )
    basis = gs.broadcast_to(gs.eye(3), batch_shape + (3, 3))

    normalized = space.metric.normal_basis(basis, base_point)

    factor = (1.0 - gs.sum(base_point**2, axis=-1)) / 2.0
    expected = factor[..., None] * basis
    assert normalized.shape == basis.shape
    assert gs.allclose(normalized, expected)
    assert gs.allclose(space.metric.squared_norm(normalized, base_point), 1.0)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1)])
def test_normal_matrix_basis_uses_the_affine_metric(batch_shape):
    """Keep matrix basis normalization and extend it to batches of SPD points."""
    space = SPDMatrices(2)
    basis = gs.array(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 3.0], [3.0, 0.0]],
            [[0.0, 0.0], [0.0, 4.0]],
        ]
    )
    basis = gs.broadcast_to(basis, batch_shape + basis.shape)
    base_point = gs.broadcast_to(gs.diag(gs.array([1.0, 2.0])), batch_shape + (1, 2, 2))

    normalized = space.metric.normal_basis(basis, base_point)

    expected = gs.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ]
    )
    assert normalized.shape == basis.shape
    assert gs.allclose(normalized, gs.broadcast_to(expected, basis.shape))
    assert gs.allclose(space.metric.squared_norm(normalized, base_point), 1.0)
