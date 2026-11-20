"""Regression tests for broadcasting n-fold exponential and logarithm inputs."""

import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import Landmarks
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.nfold_manifold import NFoldManifold, NFoldMetric

BATCH_SHAPES = [
    ((2, 1), (1, 3)),
    ((1, 3), (2, 1)),
    ((2, 1, 1), (1, 3)),
    ((), (2, 3)),
    ((2, 3), ()),
    ((1,), (1,)),
    ((), ()),
]


def _values(shape, offset):
    return offset + gs.reshape(gs.arange(math.prod(shape)), shape) / 100.0


@pytest.mark.parametrize("batch_shape,base_batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("kind", ["vector", "matrix", "nested", "landmarks"])
@pytest.mark.parametrize("method", ["exp", "log"])
def test_flat_nfold_geodesic_broadcasting(kind, method, batch_shape, base_batch_shape):
    """Preserve every broadcasted pair, including matrix and nested point axes."""
    if kind == "landmarks":
        space = Landmarks(Euclidean(2), k_landmarks=3)
    else:
        base = {
            "vector": Euclidean(2),
            "matrix": Matrices(2, 2),
            "nested": NFoldManifold(Euclidean(2), n_copies=2),
        }[kind]
        space = NFoldManifold(base, n_copies=3, equip=False)
        space.equip_with_metric(NFoldMetric, scales=gs.array([1.0, 2.0, 3.0]))
    point = _values(batch_shape + space.shape, 0.2)
    base_point = _values(base_batch_shape + space.shape, 0.1)

    result = getattr(space.metric, method)(point, base_point)

    expected = base_point + point if method == "exp" else point - base_point
    assert result.shape == expected.shape
    assert gs.allclose(result, expected)


@pytest.mark.parametrize("batch_shape,base_batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("method", ["exp", "log"])
def test_sphere_nfold_geodesic_broadcasting(method, batch_shape, base_batch_shape):
    """Check the broadcasted operations against analytic spherical formulas."""
    space = NFoldManifold(Hypersphere(2), n_copies=2)
    angle = _values(batch_shape + (2,), 0.3)
    base_angle = _values(base_batch_shape + (2,), 0.1)
    base_point = gs.stack(
        [gs.cos(base_angle), gs.sin(base_angle), gs.zeros_like(base_angle)], axis=-1
    )

    if method == "exp":
        point = angle[..., None] * gs.array([0.0, 0.0, 1.0])
        expected = gs.cos(angle)[..., None] * base_point
        expected = expected + gs.sin(angle)[..., None] * gs.array([0.0, 0.0, 1.0])
    else:
        point = gs.stack([gs.cos(angle), gs.sin(angle), gs.zeros_like(angle)], axis=-1)
        direction = gs.stack(
            [-gs.sin(base_angle), gs.cos(base_angle), gs.zeros_like(base_angle)],
            axis=-1,
        )
        expected = (angle - base_angle)[..., None] * direction

    result = getattr(space.metric, method)(point, base_point)

    assert result.shape == expected.shape
    assert gs.allclose(result, expected)
