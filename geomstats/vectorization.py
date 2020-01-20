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
        vectorized_ndim = POINT_TYPES_TO_NDIMS[point_type]
        assert ndim <= vectorized_ndim
        if ndim == vectorized_ndim:
            return False
    return True


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
            initial_ndims = []
            for param, point_type in zip(args, point_types):
                initial_ndims.append(gs.ndim(param))

                vect_param = gs.to_ndarray(
                    param, POINT_TYPES_TO_NDIMS[point_type])
                vect_args.append(vect_param)
            result = function(*vect_args, **kwargs)

            if squeeze_output_dim_0(initial_ndims, point_types):
                result = gs.squeeze(result, axis=0)
            return result
        return wrapper
    return aux_decorator
