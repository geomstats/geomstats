"""Decorator to handle vectorization.

This abstracts the backend type.
"""

import geomstats.backend as gs

POINT_TYPES = ['scalar', 'vector', 'matrix']
FLEXIBLE_TYPE = 'point'
OTHER_TYPES = ['point_type', 'else']

POINT_TYPES_TO_NDIMS = {
    'scalar': 2,
    'vector': 2,
    'matrix': 3}

ERROR_MSG = 'Invalid type: %s.'


def get_n_points(points, point_type):
    """Compute the number of points.

    Parameters
    ----------
    points : array-like
        Input points.
    point_type : str, {'scalar', ''vector', 'matrix'}
        Type of input points.

    Returns
    -------
    n_points : int
        Number of points.
    """
    n_points = 1

    points_ndim = gs.ndim(points)

    is_vect_scalar = (point_type == 'scalar' and points_ndim == 1)
    is_vect_vector = (point_type == 'vector' and points_ndim == 2)
    is_vect_matrix = (point_type == 'matrix' and points_ndim == 3)

    if is_vect_scalar or is_vect_vector or is_vect_matrix:
        n_points = gs.shape(points)[0]
    return n_points


def decorator(input_types):
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
    - gets the types of all inputs of its function:
        - args,
        - kwargs,
        - optional kwargs,
            - e.g. input_type=None,
    - gets the type of the output of its function,
        - e.g. distinguishes between 1D 'vector' vs 'scalar',
    - gets the initial shapes of all inputs of its function,
    - if needed, adapts the types of the inputs,
        - e.g. distinguishes between 'vector' or 'matrix' inputs,
        using variables 'point_type' or 'default_point_type',
    - converts the inputs into fully-vectorized inputs,
    - calls the function,
    - adapts the output shapes to match the users' expectations,
    using the initial shapes of the inputs.

    Parameters
    ----------
    input_types : list
        List of inputs' input_types, including for optional inputs.
        The `input_type`s of optional inputs will not be read
        by the decorator if the corresponding input is not given.
    """
    if not isinstance(input_types, list):
        input_types = list(input_types)

    def aux_decorator(function):
        def wrapper(*args, **kwargs):
            args_types, kwargs_types, opt_kwargs_types, is_scal = get_types(
                input_types, args, kwargs)
            args_types, kwargs_types, kwargs = adapt_types(
                args_types, kwargs_types, opt_kwargs_types, args, kwargs)
            args_kwargs_types = args_types + kwargs_types

            args_shapes = get_initial_shapes(args_types, args)
            kwargs_shapes = get_initial_shapes(kwargs_types, kwargs.values())
            initial_shapes = args_shapes + kwargs_shapes

            vect_args = vectorize_args(args_types, args)
            vect_kwargs = vectorize_kwargs(kwargs_types, kwargs)

            result = function(*vect_args, **vect_kwargs)

            result = adapt_result(
                result, initial_shapes, args_kwargs_types, is_scal)

            return result
        return wrapper
    return aux_decorator


def get_types(input_types, args, kwargs):
    """Extract the types of args, kwargs, optional kwargs and output.

    Parameters
    ----------
    input_types : list
        List of inputs' input_types, including for optional inputs.
    args : tuple
        Args of a function.
    kwargs : dict
        Kwargs of a function.

    Returns
    -------
    args_types : list
        Types of args.
    kwargs_types : list
        Types of kwargs.
    opt_kwargs_types : list
        Types of optional kwargs.
    is_scal : bool
        Boolean determining if the output is a scalar.
    """
    len_args = len(args)
    len_kwargs = len(kwargs)
    len_total = len_args + len_kwargs

    args_types = input_types[:len_args]
    kwargs_types = input_types[len_args:len_total]

    opt_kwargs_types = []
    is_scal = True
    if len(input_types) > len_total:
        opt_kwargs_types = input_types[len_total:]
        last_input_type = input_types[-1]
        if 'output_' in last_input_type:
            if last_input_type != 'output_scalar':
                is_scal = False
                opt_kwargs_types = input_types[len_total:-1]
    return (args_types, kwargs_types, opt_kwargs_types, is_scal)


def adapt_types(
        args_types, kwargs_types,
        opt_kwargs_types, args, kwargs):
    """Adapt the list of input input_types.

    Some functions are implemented with array-like arguments that can be either
    'vector' or 'matrix' depending on the value of the 'point_type'
    argument.

    This function reads the 'point_type' argument, and adapt the actual
    type of the input array-like arguments.

    Parameters
    ----------
    args_types : list
        Types of args.
    kwargs_types : list
        Types of kwargs.
    opt_kwargs_types : list
        Types of optional kwargs.
    args : tuple
        Args of a function.
    kwargs : dict
        Kwargs of a function.

    Returns
    -------
    args_types : list
        Adapted types of args.
    kwargs_types : list
        Adapted types of kwargs.
    """
    in_args = 'point_type' in args_types
    in_kwargs = 'point_type' in kwargs_types
    in_optional = 'point_type' in opt_kwargs_types

    if in_args or in_kwargs or in_optional:
        if in_args:
            i_input_type = args_types.index('point_type')
            input_type = args[i_input_type]
        elif in_kwargs:
            input_type = kwargs['point_type']

        elif in_optional:
            obj = args[0]
            input_type = obj.default_point_type
            kwargs['point_type'] = input_type
            kwargs_types.append('point_type')

        args_types = [
            input_type if pt == FLEXIBLE_TYPE else pt
            for pt in args_types]
        kwargs_types = [
            input_type if pt == FLEXIBLE_TYPE else pt
            for pt in kwargs_types]
    return args_types, kwargs_types, kwargs


def get_initial_shapes(input_types, args):
    """Extract shapes and ndims of input args or kwargs values.

    Store the shapes of the input args, or kwargs values,
    that are array-like, store None otherwise.

    Parameters
    ----------
    input_types : list
        Point types corresponding to the args, or kwargs values.
    args : tuple or dict_values
        Args, or kwargs values, of a function.

    Returns
    -------
    in_shapes : list
        Shapes of array-like input args, or kwargs values.
    """
    in_shapes = []

    for arg, input_type in zip(args, input_types):
        if input_type == 'scalar':
            arg = gs.array(arg)

        if input_type in POINT_TYPES and arg is not None:
            in_shapes.append(gs.shape(arg))
        elif input_type in OTHER_TYPES or arg is None:
            in_shapes.append(None)
        else:
            raise ValueError(ERROR_MSG % input_type)
    return in_shapes


def vectorize_args(input_types, args):
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
    input_types : list
        Point types corresponding to the args.
    args : tuple
        Args of a function.

    Returns
    -------
    vect_args : tuple
        Args of the function in their fully-vectorized form.
    """
    vect_args = []
    for arg, input_type in zip(args, input_types):
        if input_type == 'scalar':
            vect_arg = gs.to_ndarray(arg, to_ndim=1)
            vect_arg = gs.to_ndarray(vect_arg, to_ndim=2, axis=1)
        elif input_type in POINT_TYPES and arg is not None:
            vect_arg = gs.to_ndarray(
                arg, to_ndim=POINT_TYPES_TO_NDIMS[input_type])
        elif input_type in OTHER_TYPES or arg is None:
            vect_arg = arg
        else:
            raise ValueError(ERROR_MSG % input_type)
        vect_args.append(vect_arg)
    return tuple(vect_args)


def vectorize_kwargs(input_types, kwargs):
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
    input_types :list
        Point types corresponding to the args.
    kwargs : dict
        Kwargs of a function.

    Returns
    -------
    vect_kwargs : dict
        Kwargs of the function in their fully-vectorized form.
    """
    vect_kwargs = {}
    for key_arg, input_type in zip(kwargs.keys(), input_types):
        arg = kwargs[key_arg]
        if input_type == 'scalar':
            vect_arg = gs.to_ndarray(arg, to_ndim=1)
            vect_arg = gs.to_ndarray(vect_arg, to_ndim=2, axis=1)
        elif input_type in POINT_TYPES and arg is not None:
            vect_arg = gs.to_ndarray(
                arg, to_ndim=POINT_TYPES_TO_NDIMS[input_type])
        elif input_type in OTHER_TYPES or arg is None:
            vect_arg = arg
        else:
            raise ValueError(ERROR_MSG % input_type)
        vect_kwargs[key_arg] = vect_arg
    return vect_kwargs


def adapt_result(result, initial_shapes, args_kwargs_types, is_scal):
    """Adapt shape of output.

    This function squeezes the dim 0 or 1 of the output, depending on:
    - the type of the output: scalar vs else,
    - the initial shapes or args and kwargs provided by the user.

    Parameters
    ----------
    result : unspecified
        Output of the function.
    initial_shapes : list
        Shapes of args and kwargs provided by the user.
    args_kwargs_types : list
        Types of args and kwargs.
    is_scal : bool
        Boolean determining if the output 'result' is a scalar.

    Returns
    -------
    result : unspecified
        Output of the function, with adapted shape.
    """
    if squeeze_output_dim_1(
            result, initial_shapes, args_kwargs_types, is_scal):
        if result.shape[1] == 1:
            result = gs.squeeze(result, axis=1)

    if squeeze_output_dim_0(result, initial_shapes, args_kwargs_types):
        if result.shape[0] == 1:
            result = gs.squeeze(result, axis=0)

    return result


def squeeze_output_dim_0(result, in_shapes, input_types):
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
    input_types : list
        Associated list of input_type of input parameters.

    Returns
    -------
    squeeze : bool
        Boolean deciding whether to squeeze dim 0 of the output.
    """
    if isinstance(result, tuple):
        return False
    if isinstance(result, list):
        return False

    for in_shape, input_type in zip(in_shapes, input_types):
        if input_type not in POINT_TYPES:
            continue
        in_ndim = None
        if in_shape is not None:
            in_ndim = len(in_shape)
        if in_ndim is not None:
            vect_ndim = POINT_TYPES_TO_NDIMS[input_type]
            if in_ndim > vect_ndim:
                raise ValueError(
                    'Fully-vectorizing an input can only increase its ndim.')
            if in_ndim == vect_ndim:
                return False
    return True


def squeeze_output_dim_1(result, in_shapes, input_types, is_scal=True):
    """Determine if the output needs to be squeezed on dim 1.

    This happens if the user represents scalars as array of shapes:
    [n_samples,] instead of [n_samples, 1]
    Dimension 1 is squeezed by default if input_type is 'scalar'.
    Dimension 1 is not squeezed if the user inputs at least one scalar with
    a singleton in dimension 1.

    Parameters
    ----------
    result: array-like
        Result output by the function, before reshaping.
    in_shapes : list
        Initial shapes of input parameters, as entered by the user.
    input_types : list
        Associated list of input_type of input parameters.

    Returns
    -------
    squeeze : bool
        Boolean deciding whether to squeeze dim 1 of the output.
    """
    if not is_scal:
        return False
    if not is_scalar(result):
        return False

    for shape, input_type in zip(in_shapes, input_types):
        if input_type == 'scalar':
            ndim = len(shape)
            if ndim > 2:
                raise ValueError('The ndim of a scalar cannot be > 2.')
            if ndim == 2:
                return False
    return True


def is_scalar(vect_array):
    """Test if a "fully-vectorized" array represents a scalar.

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
