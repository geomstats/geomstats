"""Decorator to handle vectorization.

This abstracts the backend type.
This assumes that functions are implemented to return vectorized outputs.
"""

import geomstats.backend as gs

POINT_TYPES_TO_NDIMS = {
    'scalar': 2,
    'vector': 2,
    'matrix': 3}


def decorator(point_types):
    """Vectorize geomstats functions.

    This decorator assumes that its function:
    - works with fully-vectorized inputs,
    - returns fully-vectorized outputs,

    where "fully-vectorized" means that:
    - one scalar has shape [1, 1],
    - n scalars have shape [n, 1],
    - one d-D vector has shape [1, d],
    - n d-D vectors have shape [n, d],etc
    etc.

    The decorator:
    - converts the inputs into fully-vectorized inputs,
    - calls the function,
    - adapts the output shapes to match the users' expectations.

    Parameters
    ----------
    point_types : list
        List of inputs' point_types, including for optional inputs.
        The `point_type`s of optional inputs will not be read
        by the decorator if the corresponding input is not given.
    """
    if not isinstance(point_types, list):
        point_types = list(point_types)

    def aux_decorator(function):
        def wrapper(*args, **kwargs):

            # print('\nbefore')
            # print('args_types')
            # print(args_types)
            # print('kwargs_types')
            # print(kwargs_types)
            # print('opt_kwargs_types')
            # print(opt_kwargs_types)
            # print('args')
            # print(args)
            # print('kwargs')
            # print(kwargs)
            args_types, kwargs_types, opt_kwargs_types, scal_res = get_types(
                point_types, args, kwargs)

            args_types, kwargs_types = adapt_point_types(
                args_types, kwargs_types, opt_kwargs_types, args, kwargs)

            # print('\nafter')
            # print('args_types')
            # print(args_types)
            # print('kwargs_types')
            # print(kwargs_types)
            # print('args')
            # print(args)
            # print('kwargs')
            # print(kwargs)

            args_shapes = initial_shapes(args_types, args)
            kwargs_shapes = initial_shapes(kwargs_types, kwargs.values())
            in_shapes = args_shapes + kwargs_shapes

            vect_args = vectorize_args(args_types, args)
            vect_kwargs = vectorize_kwargs(kwargs_types, kwargs)

            result = function(*vect_args, **vect_kwargs)

            adapted_point_types = args_types + kwargs_types

            if squeeze_output_dim_1(result,
                                    in_shapes,
                                    adapted_point_types, scal_res):
                if result.shape[1] == 1:
                    result = gs.squeeze(result, axis=1)

            if squeeze_output_dim_0(result, in_shapes, adapted_point_types):
                if result.shape[0] == 1:
                    result = gs.squeeze(result, axis=0)
            return result
        return wrapper
    return aux_decorator


def get_types(point_types, args, kwargs):
    """Extract the types of args, kwargs, optional kwargs and output.

    Parameters
    ----------
    point_types : list
    args : tuple
    kwargs : dict

    Returns
    -------
    args_types :
    kwargs_types :
    opt_kwargs_types :
    scal_res :
    """
    len_args = len(args)
    len_kwargs = len(kwargs)
    len_total = len_args + len_kwargs

    args_types = point_types[:len_args]
    kwargs_types = point_types[len_args:len_total]

    opt_kwargs_types = []
    scal_res = True
    if len(point_types) > len_total:
        opt_kwargs_types = point_types[len_total:]
        if point_types[-1] == 'no_scalar_result':
            scal_res = False
            opt_kwargs_types = point_types[len_total:-1]
    return (args_types, kwargs_types, opt_kwargs_types, scal_res)


def squeeze_output_dim_0(result, in_shapes, point_types):
    """Determine if the output needs to be squeezed on dim 0.

    The dimension 0 is squeezed iff all input parameters:
    - contain one sample,
    - have the corresponding dimension 0 squeezed,
    i.e. if all input parameters have ndim strictly less than the ndim
    corresponding to their vectorized shape.

    Parameters
    ----------
    in_ndims : list
        Initial ndims of input parameters, as entered by the user.
    point_types : list
        Associated list of point_type of input parameters.

    Returns
    -------
    squeeze : bool
        Boolean deciding whether to squeeze dim 0 of the output.
    """
    if isinstance(result, tuple):
        return False
    if isinstance(result, list):
        return False

    for in_shape, point_type in zip(in_shapes, point_types):
        in_ndim = None
        if point_type not in ['scalar', 'vector', 'matrix']:
            continue
        if in_shape is not None:
            in_ndim = len(in_shape)
        if in_ndim is not None:
            vect_ndim = POINT_TYPES_TO_NDIMS[point_type]
            assert in_ndim <= vect_ndim
            if in_ndim == vect_ndim:
                return False
    return True


def is_scalar(vect_array):
    """Test if an array represents a scalar.

    Parameters
    ----------
    vect_array :  array-like
        Array to be tested.

    Returns
    -------
    is_scalar : bool
        Boolean determining if vect_array is a fully-vectorized scalar.
    """
    if isinstance(vect_array, tuple):
        return False
    if isinstance(vect_array, list):
        return False

    has_ndim_2 = vect_array.ndim == 2
    if not has_ndim_2:
        return False
    has_singleton_dim_1 = vect_array.shape[1] == 1
    return has_singleton_dim_1


def squeeze_output_dim_1(result, in_shapes, point_types, scalar_result=True):
    """Determine if the output needs to be squeezed on dim 1.

    This happens if the user represents scalars as array of shapes:
    [n_samples,] instead of [n_samples, 1]
    Dimension 1 is squeezed by default if point_type is 'scalar'.
    Dimension 1 is not squeezed if the user inputs at least one scalar with
    a singleton in dimension 1.

    Parameters
    ----------
    result: array-like
        Result output by the function, before reshaping.
    in_shapes : list
        Initial shapes of input parameters, as entered by the user.
    point_types : list
        Associated list of point_type of input parameters.

    Returns
    -------
    squeeze : bool
        Boolean deciding whether to squeeze dim 1 of the output.
    """
    if not scalar_result:
        return False
    if not is_scalar(result):
        return False

    for shape, point_type in zip(in_shapes, point_types):
        if point_type == 'scalar':
            ndim = len(shape)
            assert ndim <= 2
            if ndim == 2:
                return False
    return True


def adapt_point_types(
        args_types, kwargs_types,
        opt_kwargs_types, args, kwargs):
    """Adapt the list of input point_types."""
    in_args = 'point_type' in args_types
    in_kwargs = 'point_type' in kwargs_types
    in_optional = 'point_type' in opt_kwargs_types

    if in_args or in_kwargs or in_optional:
        if in_args:
            i_point_type = args_types.index('point_type')
            point_type = args[i_point_type]
        elif in_kwargs:
            point_type = kwargs['point_type']

        elif in_optional:
            obj = args[0]
            point_type = obj.default_point_type

        args_types = [
            pt if pt != 'point' else point_type for pt in args_types]
        kwargs_types = [
            pt if pt != 'point' else point_type for pt in kwargs_types]
    return args_types, kwargs_types


def initial_shapes(point_types, args):
    """Extract shapes and ndims of input args or kwargs values.

    Store the shapes of the input args, or kwargs values,
    that are array-like, store None otherwise.

    Parameters
    ----------
    point_types : list
        Point types corresponding to the args, or kwargs values.
    args : tuple or dict_values
        Args, or kwargs values, of a function.

    Returns
    -------
    in_shapes : list
        Shapes of array-like input args, or kwargs values.
    """
    in_shapes = []

    for i_arg, arg in enumerate(args):
        point_type = point_types[i_arg]

        if point_type == 'scalar':
            arg = gs.array(arg)

        if point_type == 'else' or arg is None:
            in_shapes.append(None)
        else:
            in_shapes.append(gs.shape(arg))
    return in_shapes


def vectorize_args(point_types, args):
    """Vectorize input args.

    Transform input array-like args into their fully-vectorized form,

    where "fully-vectorized" means that:
    - one scalar has shape [1, 1],
    - n scalars have shape [n, 1],
    - one d-D vector has shape [1, d],
    - n d-D vectors have shape [n, d],
    etc.

    Parameters
    ----------
    point_types : list
        Point types corresponding to the args.
    args : tuple
        Args of a function.

    Returns
    -------
    vect_args : tuple
        Args of the function in their fully-vectorized form.
    """
    vect_args = []
    for i_arg, arg in enumerate(args):
        point_type = point_types[i_arg]
        if point_type in ['else', 'point_type'] or arg is None:
            vect_arg = arg
        elif point_type == 'scalar':
            vect_arg = gs.to_ndarray(arg, to_ndim=1)
            vect_arg = gs.to_ndarray(vect_arg, to_ndim=2, axis=1)
        elif point_type in ['vector', 'matrix']:
            vect_arg = gs.to_ndarray(
                arg, to_ndim=POINT_TYPES_TO_NDIMS[point_type])
        else:
            raise ValueError('Invalid point type: %s' % point_type)
        vect_args.append(vect_arg)
    return tuple(vect_args)


def vectorize_kwargs(point_types, kwargs):
    """Vectorize input kwargs.

    Transform input array-like kwargs into their fully-vectorized form,

    where "fully-vectorized" means that:
    - one scalar has shape [1, 1],
    - n scalars have shape [n, 1],
    - one d-D vector has shape [1, d],
    - n d-D vectors have shape [n, d],
    etc.

    Parameters
    ----------
    point_types :list
        Point types corresponding to the args.
    kwargs : dict
        Kwargs of a function.

    Returns
    -------
    vect_kwargs : dict
        Kwargs of the function in their fully-vectorized form.
    """
    vect_kwargs = {}
    for i_arg, key_arg in enumerate(kwargs.keys()):
        point_type = point_types[i_arg]
        arg = kwargs[key_arg]
        if point_type in ['else', 'point_type'] or arg is None:
            vect_arg = arg
        elif point_type == 'scalar':
            vect_arg = gs.to_ndarray(arg, to_ndim=1)
            vect_arg = gs.to_ndarray(vect_arg, to_ndim=2, axis=1)
        elif point_type in ['vector', 'matrix']:
            vect_arg = gs.to_ndarray(
                arg, to_ndim=POINT_TYPES_TO_NDIMS[point_type])
        else:
            raise ValueError('Invalid point type.')
        vect_kwargs[key_arg] = vect_arg
    return vect_kwargs
