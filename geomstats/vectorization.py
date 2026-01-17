"""Decorator to handle vectorization.

This abstracts the backend type.
"""

import math

import geomstats.backend as gs
from geomstats.geometry.stratified.point_set import Point


def _get_max_ndim_point(*point):
    """Identify point with higher dimension.

    Parameters
    ----------
    point : array-like

    Returns
    -------
    max_ndim_point : array-like
        Point with higher dimension.
    """
    max_ndim_point = point[0]
    for point_ in point[1:]:
        if point_.ndim > max_ndim_point.ndim:
            max_ndim_point = point_

    return max_ndim_point


def get_n_points(point_ndim, *point):
    """Compute the number of points.

    Parameters
    ----------
    point_ndim : int
        Point number of array dimensions.
    point : array-like
        Point belonging to the space.

    Returns
    -------
    n_points : int
        Number of points.
    """
    point_max_ndim = _get_max_ndim_point(*point)
    return math.prod(point_max_ndim.shape[:-point_ndim])


def check_is_batch(point_ndim, *point):
    """Check if inputs are batch.

    Parameters
    ----------
    point_ndim : int
        Point number of array dimensions.
    point : array-like
        Point belonging to the space.

    Returns
    -------
    is_batch : bool
        Returns True if point contains several points.
    """
    return any(point_.ndim > point_ndim for point_ in point)


def get_batch_shape(point_ndim, *point):
    """Get batch shape.

    Parameters
    ----------
    point_ndim : int
        Point number of array dimensions.
    point : array-like or None
        Point belonging to the space.

    Returns
    -------
    batch_shape : tuple
        Returns the shape related with batch. () if only one point.
    """
    point = list(filter(_is_not_none, point))
    if len(point) == 0:
        return ()
    point_max_ndim = _get_max_ndim_point(*point)
    return point_max_ndim.shape[:-point_ndim]


def repeat_point(point, n_reps=2, expand=False):
    """Repeat point.

    Parameters
    ----------
    point : array-like or Point
        Point of a space.
    n_reps : int
        Number of times the point should be repeated.
    expand : bool
        Repeat even if n_reps == 1.

    Returns
    -------
    rep_point : array-like or list[Point]
        point repeated n_reps times.
    """
    if isinstance(point, Point):
        if not expand and n_reps == 1:
            return point

        return [point] * n_reps

    if not expand and n_reps == 1:
        return gs.copy(point)

    return gs.repeat(gs.expand_dims(point, 0), n_reps, axis=0)


def _is_not_none(value):
    """Check if a value is None."""
    return value is not None


def repeat_out(point_ndim, out, *point, out_shape=()):
    """Repeat out shape after finding batch shape.

    Parameters
    ----------
    point_ndim : int
        Point number of array dimensions.
    out : array-like
        Output to be repeated
    point : array-like or None
        Point belonging to the space.
    out_shape : tuple
        Indicates out shape for no batch computations.

    Returns
    -------
    out : array-like
        If no batch, then input is returned. Otherwise it is broadcasted.
    """
    point = filter(_is_not_none, point)
    batch_shape = get_batch_shape(point_ndim, *point)
    if out.shape[: -len(out_shape)] != batch_shape:
        return gs.broadcast_to(out, batch_shape + out_shape)
    return out


def repeat_out_multiple_ndim(
    out, point_ndim_1, points_1, point_ndim_2, points_2, out_ndim=0
):
    """Repeat out after finding batch shape.

    Differs from `repeat_out` by accepting two sets of point_ndim arrays.

    Parameters
    ----------
    out : array-like
        Output to be repeated
    point_ndim_1 : int
        Point number of array dimensions.
    points_1 : tuple[array-like or None]
        Arrays of dimension point_ndim_1 or higher.
    point_ndim_2 : int
        Point number of array dimensions.
    points_2 : tuple[array-like or None]
        Arrays of dimension point_ndim_2 or higher.
    out_ndim : int
        Out number of array dimensions.

    Returns
    -------
    out : array-like
        If no batch, then input is returned. Otherwise it is broadcasted.
    """
    batch_shape = get_batch_shape(point_ndim_1, *points_1)
    if not batch_shape:
        batch_shape = get_batch_shape(point_ndim_2, *points_2)

    out_shape = out.shape[-out_ndim:]
    if out.shape[:-out_ndim] != batch_shape:
        return gs.broadcast_to(out, batch_shape + out_shape)
    return out


def broadcast_to_multibatch(batch_shape_a, batch_shape_b, array_a, *array_b):
    """Broadcast to multibatch.

    Gives to both arrays batch shape `batch_shape_b + batch_shape_a`.

    Does nothing if one of the batch shapes is empty.

    Parameters
    ----------
    batch_shape_a : tuple
        Batch shape of array_a.
    batch_shape_b : tuple
        Batch shape of array_b.
    array_a : array
    array_b : array
    """
    multi_b = len(array_b) > 1

    if not batch_shape_a or not batch_shape_b:
        return (array_a, array_b) if multi_b else (array_a, array_b[0])

    array_a_ = gs.broadcast_to(array_a, batch_shape_b + array_a.shape)

    n_batch_b = len(batch_shape_b)
    indices_in = list(range(len(batch_shape_a)))
    indices_out = [index + n_batch_b for index in indices_in]

    array_b_ = []
    for array in array_b:
        array_b_aux = gs.broadcast_to(array, batch_shape_a + array.shape)
        array_b_.append(gs.moveaxis(array_b_aux, indices_in, indices_out))

    return (array_a_, array_b_) if multi_b else (array_a_, array_b_[0])
