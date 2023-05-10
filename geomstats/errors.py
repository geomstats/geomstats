"""Checks and associated errors."""

import math

import geomstats.backend as gs


def check_integer(n, n_name):
    """Raise an error if n is not a > 0 integer.

    Parameters
    ----------
    n : unspecified
       Parameter to be tested.
    n_name : string
       Name of the parameter.
    """
    if n is not None and not (isinstance(n, int) and n > 0) and n != math.inf:
        raise ValueError(
            f"{n_name} is required to be either"
            " None, math.inf or a strictly positive integer,"
            f" got {n}."
        )


def check_positive(param, param_name):
    """Raise an error if param is not a > 0 number.

    Parameters
    ----------
    param : unspecified
       Parameter to be tested.
    param_name : string
       Name of the parameter.
    """
    if not (
        (isinstance(param, (int, float)) or (gs.is_array(param) and param.ndim == 0))
        and param > 0
    ):
        raise ValueError(f"{param_name} must be positive.")


def check_belongs(point, manifold, **kwargs):
    """Raise an error if point does not belong to the input manifold.

    Parameters
    ----------
    point : array-like
        Point to be tested.
    manifold : Manifold
        Manifold to which the point should belong.
    manifold_name : string
        Name of the manifold for the error message.
    """
    if not gs.all(manifold.belongs(point, **kwargs)):
        raise RuntimeError(
            f"Some points do not belong to manifold '{type(manifold).__name__}'"
            f" of dimension {manifold.dim}."
        )


def check_parameter_accepted_values(param, param_name, accepted_values):
    """Raise an error if parameter does not belong to a set of values.

    Parameters
    ----------
    param : unspecified
        Parameter to be tested.
    param_name : string
        Name of the parameter.
    accepted_values : list
        Accepted values that the parameter can take.
    """
    if param not in accepted_values:
        raise ValueError(
            f"Parameter {param_name} needs to be in {accepted_values}, got: {param}."
        )


def check_point_shape(point, manifold, suppress_error=False):
    """Check if the shape of point does not match the shape of a manifold or metric.

    If the final elements of the shape of point do not match the shape of manifold
    (which may be any object with a shape attribute, such as a Riemannian metric) then
    point cannot be an array of points on the manifold (or similar) and a ValueError is
    raised. The error can be suppressed by setting suppress_error to True.

    Parameters
    ----------
    point : array-like
        The point to check the shape of.
    manifold : {Manifold, RiemannianMetric}
        The object to check the point against
    suppress_error : bool
        Whether to suppress the ShapeError if the shapes do not match. Optional, default
        is False.

    Returns
    -------
    shapes_match : bool
        Whether the shape of the point matches the shape of the manifold or metric.

    Raises
    ------
    ValueError
        If the final dimensions of point are not equal to the final dimensions of
        manifold.
    """
    representation_type = -1 * len(manifold.shape)
    shapes_match = (
        point.shape[representation_type:] == manifold.shape[representation_type:]
    )
    if not suppress_error and not shapes_match:
        shape_error_msg = (
            f"The shape of {point}, which is {point.shape}, is not"
            f" compatible with the shape of the {type(manifold).__name__}"
            f" object, which is {manifold.shape}."
        )
        raise ShapeError(shape_error_msg)
    return shapes_match


class ShapeError(ValueError):
    """Raised when there is an incompatibility between shapes."""
