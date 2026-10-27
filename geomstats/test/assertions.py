"""Geometric assertions for tests."""

import numpy as np

import geomstats.backend as gs
from geomstats.test.test_case import assert_true, autograd_backend, np_backend
from geomstats.vectorization import get_batch_shape


def _assert_all_equal(a, b):
    if np_backend() or autograd_backend():
        np.testing.assert_array_equal(a, b)
    else:
        assert_true(gs.all(gs.equal(a, b)))


def _assert_all_true(res, point_ndim, *points):
    expected = gs.ones(get_batch_shape(point_ndim, *points), dtype=bool)
    _assert_all_equal(res, expected)


def assert_belongs(space, point, atol=gs.atol):
    """Assert point belongs to space.

    Parameters
    ----------
    space : Manifold
        Space.
    point : array-like, shape=[..., *space.shape]
        Point.
    atol : float
        Absolute tolerance.
    """
    res = space.belongs(point, atol=atol)
    _assert_all_true(res, space.point_ndim, point)


def assert_is_tangent(space, vector, base_point, atol=gs.atol):
    """Assert vector is tangent to space at base point.

    Parameters
    ----------
    space : Manifold
        Space.
    vector : array-like, shape=[..., *space.shape]
        Vector.
    base_point : array-like, shape=[..., *space.shape]
        Point on the manifold.
    atol : float
        Absolute tolerance.
    """
    res = space.is_tangent(vector, base_point, atol=atol)
    _assert_all_true(res, space.point_ndim, vector, base_point)


def assert_is_horizontal(total_space, tangent_vec, base_point, atol=gs.atol):
    """Assert tangent vector is horizontal at base point.

    Parameters
    ----------
    total_space : Manifold
        Total space equipped with a fiber bundle.
    tangent_vec : array-like, shape=[..., *total_space.shape]
        Tangent vector.
    base_point : array-like, shape=[..., *total_space.shape]
        Point on the total space.
    atol : float
        Absolute tolerance.
    """
    res = total_space.fiber_bundle.is_horizontal(tangent_vec, base_point, atol=atol)
    _assert_all_true(res, total_space.point_ndim, tangent_vec, base_point)


def assert_is_vertical(total_space, tangent_vec, base_point, atol=gs.atol):
    """Assert tangent vector is vertical at base point.

    Parameters
    ----------
    total_space : Manifold
        Total space equipped with a fiber bundle.
    tangent_vec : array-like, shape=[..., *total_space.shape]
        Tangent vector.
    base_point : array-like, shape=[..., *total_space.shape]
        Point on the total space.
    atol : float
        Absolute tolerance.
    """
    res = total_space.fiber_bundle.is_vertical(tangent_vec, base_point, atol=atol)
    _assert_all_true(res, total_space.point_ndim, tangent_vec, base_point)
