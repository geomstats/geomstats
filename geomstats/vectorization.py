"""
Decorator to handle vectorization.

This abstracts the backend type.
"""

import geomstats.backends as gs

POINT_TYPES_TO_NDIMS = {
    'vector': 2,
    'matrix': 3}


def final_shape(initial_shapes, point_types):
    """Compute the desired output shape."""
    raise NotImplementedError()


def decorator_factory(param_names, point_types):
    """Decorator vectorizing geomstats functions.

    Parameters
    ----------
    param_names: list
        parameters names to be vectorized
    point_types: list
        associated list of ndims after vectorization
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            vectorized_kwargs = {}
            initial_shapes = []
            for name, point_type in zip(param_names, point_types):
                param = kwargs.get(name, None)
                initial_shapes.append(gs.shape(param))

                param = gs.to_ndim(param, POINT_TYPES_TO_NDIMS[point_type])
                vectorized_kwargs[name] = param

            result = function(*args, **vectorized_kwargs)

            result_shape = final_shape(initial_shapes, point_types)
            return gs.reshape(result, result_shape)
        return wrapper
    return decorator
