"""Finite differences machinery."""

import geomstats.backend as gs


def forward_difference(array, delta=None, axis=-1):
    """Forward difference in a Euclidean space.

    Points live in R^m, but are a k-dim embedding (e.g. a curve).
    Assumes points are in correspondence for cases higher than dim=1.

    Parameters
    ----------
    array : array-like
        Values of a function.
    delta : float
        Spacing between points.
    axis : int
        Axis in which perform the difference.
        Must be given backwards.

    Returns
    -------
    forward_diff : array-like
        Shape in the specified axis reduces by one.
    """
    n = array.shape[axis]
    if delta is None:
        delta = 1 / n

    point_ndim_slc = (slice(None),) * (abs(axis) - 1)

    slc = (..., slice(1, n)) + point_ndim_slc
    forward = array[slc]

    slc = (..., slice(0, n - 1)) + point_ndim_slc
    center = array[slc]
    return (forward - center) / delta


def centered_difference(array, delta=None, axis=-1, endpoints=False):
    """Centered difference in a Euclidean space.

    Points live in R^m, but are a k-dim embedding (e.g. a curve).
    Assumes points are in correspondence for cases higher than dim=1.

    Parameters
    ----------
    array : array-like
        Values of a function.
    delta : float
        Spacing between points.
    axis : int
        Axis in which perform the difference.
        Must be given backwards.
    endpoints : bool
        If True, endpoints are computed by backward and forward differences,
        respectively.

    Returns
    -------
    centered_diff : array-like
        Same shape as array.
    """
    n = array.shape[axis]
    if delta is None:
        delta = 1 / n

    point_ndim_slc = (slice(None),) * (abs(axis) - 1)

    slc = (..., slice(2, n)) + point_ndim_slc
    forward = array[slc]

    slc = (..., slice(0, n - 2)) + point_ndim_slc
    backward = array[slc]
    diff = (forward - backward) / (2 * delta)

    if endpoints:
        slc_left = (..., [0]) + point_ndim_slc
        slc_left_forward = (..., [1]) + point_ndim_slc
        diff_left = (array[slc_left_forward] - array[slc_left]) / delta

        slc_right = (..., [-1]) + point_ndim_slc
        slc_right_backward = (..., [-2]) + point_ndim_slc
        diff_right = (array[slc_right] - array[slc_right_backward]) / delta

        slc_right = (..., [-1]) + point_ndim_slc
        return gs.concatenate((diff_left, diff, diff_right), axis=axis)

    return diff


def second_centered_difference(array, delta=None, axis=-1):
    """Second centered difference in a Euclidean space.

    Points live in R^m, but are a k-dim embedding (e.g. a curve).
    Assumes points are in correspondence for cases higher than dim=1.

    Parameters
    ----------
    array : array-like
        Values of a function.
    delta : float
        Spacing between points.
    axis : int
        Axis in which perform the difference.
        Must be given backwards.

    Returns
    -------
    second_centered_diff : array-like
        Shape in the specified axis reduces by two (endpoints).
    """
    n = array.shape[axis]
    if delta is None:
        delta = 1 / n

    point_ndim_slc = (slice(None),) * (abs(axis) - 1)

    slc = (..., slice(2, n)) + point_ndim_slc
    forward = array[slc]

    slc = (..., slice(0, n - 2)) + point_ndim_slc
    backward = array[slc]

    slc = (..., slice(1, n - 1)) + point_ndim_slc
    central = array[slc]

    return (forward + backward - 2 * central) / (delta**2)
