"""
Decorator to handle vectorization.

This abstracts the backend type.
"""

import geomstats.backend as gs

POINT_TYPES_TO_NDIMS = {
    'scalar': 2,
    'vector': 2,
    'matrix': 3}


def squeeze_output_dim_0(initial_ndims, point_types):
    """Determine if the output needs to squeeze a singular dimension 0.

    The dimension 0 is squeezed iff all input parameters:
    - contain one sample,
    - have the corresponding dimension 0 squeezed,
    i.e. if all input parameters have ndim strictly less than the ndim
    corresponding to their vectorized shape.
    """
    for ndim, point_type in zip(initial_ndims, point_types):
        vect_ndim = POINT_TYPES_TO_NDIMS[point_type]
        assert ndim <= vect_ndim
        if ndim == vect_ndim:
            return False
    return True


def is_scalar(vect_array):
    has_ndim_2 = vect_array.ndim == 2
    has_singleton_dim_1 = vect_array.shape[1] == 1
    return has_ndim_2 and has_singleton_dim_1


def squeeze_output_dim_1(result, initial_shapes, point_types):
    """Determine if the output needs to squeeze a singular dimension 1.

    This happens if the user represents scalars as array of shapes:
    [n_samples,] instead of [n_samples, 1]

    The dimension 1 is squeezed iff:
    - the return point type is a scalar,
    - all input scalars have squeezed dimension 1.
    """
    if not is_scalar(result):
        return False

    for shape, point_type in zip(initial_shapes, point_types):
        ndim = len(shape)
        if point_type == 'scalar':
            assert ndim <= 2
            if ndim == 2:
                return False
            if ndim == 0:
                return True
            if ndim == 1:
                if shape[0] > 1:
                    return True
    return False


def decorator(param_names, point_types):
    """Decorator vectorizing geomstats functions.

    Parameters
    ----------
    param_names: list
        parameters names to be vectorized
    point_types: list
        associated list of ndims after vectorization
    """
    def aux_decorator(function):
        def wrapper(*args, **kwargs):
            vect_args = []
            initial_shapes = []
            initial_ndims = []
            for param, point_type in zip(args, point_types):
                initial_shapes.append(param.shape)
                initial_ndims.append(gs.ndim(param))

                vect_param = gs.to_ndarray(
                    param, POINT_TYPES_TO_NDIMS[point_type])
                vect_args.append(vect_param)
            result = function(*vect_args, **kwargs)

            if squeeze_output_dim_1(result, initial_shapes, point_types):
                result = gs.squeeze(result, axis=1)

            if squeeze_output_dim_0(initial_ndims, point_types):
                result = gs.squeeze(result, axis=0)
            return result
        return wrapper
    return aux_decorator
