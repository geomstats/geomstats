"""Decorator to handle vectorization."""

import functools

import geomstats.backend as gs
from geomstats.geometry.stratified.point_set import PointBatch


def broadcast_lists(*lists):
    """Broadcast lists.

    Similar behavior as ``gs.broadcast_arrays``, but for lists.

    Parameters
    ----------
    *lists : list

    Returns
    -------
    *broadcasted_lists : list
        Lists with broadcasted length.
    """
    lens = [len(list_) for list_ in lists]
    n_max = max(lens)
    if n_max == 1:
        return lists

    out = []
    for len_, list_ in zip(lens, lists):
        if len_ == 1:
            out.append([list_[0]] * n_max)

        elif len_ == n_max:
            out.append(list_)

        else:
            raise Exception(f"Cannot broadcast lens: {lens}")

    return out


def _manipulate_input(arg):
    if not isinstance(arg, (list, tuple)):
        return [arg], True

    return arg, False


def _manipulate_output_iterable(out):
    return PointBatch(out)


def _manipulate_output(
    out, to_list, manipulate_output_iterable=_manipulate_output_iterable
):
    is_array = gs.is_array(out)
    is_iterable = isinstance(out, (list, tuple))

    if not (gs.is_array(out) or is_iterable):
        return out

    if to_list:
        if is_array:
            return gs.array(out[0])
        if is_iterable:
            return out[0]

    if is_iterable:
        return manipulate_output_iterable(out)

    return out


def vectorize_point(
    *args_positions,
    manipulate_input=_manipulate_input,
    manipulate_output=_manipulate_output,
):
    """Check point type and transform in iterable if not the case.

    Parameters
    ----------
    args_positions : tuple
        Position and corresponding argument name. A tuple for each position.

    Notes
    -----
    Explicitly defining args_positions and args names ensures it works for all
    combinations of input calling.
    """

    def _dec(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            to_list = True
            args = list(args)
            for pos, name in args_positions:
                if name in kwargs:
                    kwargs[name], to_list_ = manipulate_input(kwargs[name])
                else:
                    args[pos], to_list_ = manipulate_input(args[pos])

                to_list = to_list and to_list_

            out = func(*args, **kwargs)

            return manipulate_output(out, to_list)

        return _wrapped

    return _dec
