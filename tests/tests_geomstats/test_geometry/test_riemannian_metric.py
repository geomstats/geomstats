from math import prod

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices


@pytest.fixture(autouse=True)
def fixed_random_samples(monkeypatch):
    def sample(*shape):
        return gs.reshape(
            gs.arange(1, prod(shape) + 1, dtype=gs.get_default_dtype()), shape
        )

    monkeypatch.setattr(gs.random, "rand", sample)


@pytest.mark.parametrize("n_vectors", [1, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_random_unit_tangent_vec_on_sphere(n_vectors, batched):
    sphere = Hypersphere(2)
    point = gs.array([0.0, 0.0, 1.0])
    if batched:
        point = gs.stack([point, gs.array([1.0, 0.0, 0.0])])
    if batched and n_vectors > 1:
        with pytest.raises(ValueError, match="Several tangent vectors"):
            sphere.metric.random_unit_tangent_vec(point, n_vectors)
        return
    vectors = sphere.metric.random_unit_tangent_vec(point, n_vectors)
    expected_shape = (2, 3) if batched else ((4, 3) if n_vectors == 4 else (3,))
    assert vectors.shape == expected_shape
    assert gs.all(sphere.is_tangent(vectors, point))
    assert gs.allclose(sphere.metric.norm(vectors, point), 1.0)


def test_random_unit_tangent_vec_is_symmetric_for_spd():
    space = SPDMatrices(2)
    point = gs.array([[2.0, 0.0], [0.0, 1.0]])
    vectors = space.metric.random_unit_tangent_vec(point, n_vectors=3)
    assert vectors.shape == (3, 2, 2)
    assert gs.all(space.is_tangent(vectors, point))
    assert gs.allclose(space.metric.norm(vectors, point), 1.0)


def test_random_unit_tangent_vec_euclidean():
    space = Euclidean(3)
    vectors = space.metric.random_unit_tangent_vec(gs.zeros(3), n_vectors=3)
    assert vectors.shape == (3, 3)
    assert gs.allclose(gs.linalg.norm(vectors, axis=-1), 1.0)


def test_random_unit_tangent_vec_intrinsic_circle():
    circle = Hypersphere(1, intrinsic=True)
    point = gs.array([1.0])
    vectors = circle.metric.random_unit_tangent_vec(point, n_vectors=3)
    assert vectors.shape == (3, 1)
    assert gs.allclose(circle.metric.norm(vectors, point), 1.0)
