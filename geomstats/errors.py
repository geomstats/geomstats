"""Checks and associated errors."""

import math
import os

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
    if not (isinstance(n, int) and n > 0):
        if n is not None and n != math.inf:
            raise ValueError(
                f"{n_name} is required to be either"
                f" None, math.inf or a strictly positive integer,"
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
    if not (isinstance(param, (int, float)) and param > 0):
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
            f" of dimension {manifold.dim}.")


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


def check_tf_error(exception, name):
    """Raise error in tensorflow."""
    if os.environ["GEOMSTATS_BACKEND"] == "tensorflow":
        from tensorflow import errors

        return getattr(errors, name)
    return exception


def check_point_shape(point, manifold):
    """Raise an error if the shape of point does not match the shape of a manifold.

    If the final elements of the shape of point do not match the shape of manifold
    (which may be any object with a shape attribute, such as a Riemannian metric, then
    point cannot be an array of points on the manifold (or similar) and a ValueError is
    raised.

    Parameters
    ----------
    point : array-like
        The point to check the shape of.
    manifold : {Manifold, RiemannianMetric}
        The object to check the point against

    Raises
    ------
    ValueError
        If the final dimensions of point are not equal to the final dimensions of
        manifold.
    """
    shape_error_msg = (f"The shape of {point}, which is {point.shape}, is not"
                       f" compatible with the shape of the {type(manifold).__name__}"
                       f" object, which is {manifold.shape}.")
    representation_type = -1 * len(manifold.shape)
    if point.shape[representation_type:] != manifold.shape[representation_type:]:
        raise ShapeError(shape_error_msg)


class ShapeError(ValueError):
    """Raised when there is an incompatibility between shapes."""
