"""Scalar product of a Riemannian metric.

Define the product of a Riemannian metric with a scalar number.

Public Methods:
    register_scaled_method(func_name, scaling_type)

Public classes
    ScalarProductMetric

Lead author: John Harvey.
"""

from functools import wraps

import geomstats.backend as gs
import geomstats.errors


def register_scaled_method(func_name, scaling_type):
    """Register the scaling factor of a method of a RiemannianMetric.

    The ScalarProductMetric class rescales various methods of a
    RiemannianMetric by the correct factor. The default behaviour is to
    rescale linearly. This method allows the user to add a new method to be
    rescaled according to a different rule.

    Note that this method must be called before the ScalarProductMetric is
    instantiated. It does not affect objects which already exist.

    Parameters
    ----------
    func_name : str
        The name of a method from a RiemannianMetric object which must be
        rescaled.
    scaling_type : str, {'sqrt',
                         'linear',
                         'quadratic',
                         'inverse',
                         'inverse_sqrt'}
        How the method should be rescaled as a function of
        ScalarProductMetric.scale.

    """
    _ScaledMethodsRegistry._add_scaled_method(func_name, scaling_type)


def _wrap_attr(scaling_factor, func):
    @wraps(func)
    def response(*args, **kwargs):
        res = scaling_factor * func(*args, **kwargs)
        return res

    return response


class _ScaledMethodsRegistry:
    """Class to hold lists of methods and their scaling functions."""

    _SQRT_LIST = ["norm", "dist", "dist_broadcast", "dist_pairwise", "diameter"]
    _LINEAR_LIST = [
        "metric_matrix",
        "inner_product",
        "inner_product_derivative_matrix",
        "squared_norm",
        "squared_dist",
        "covariant_riemann_tensor",
    ]
    _QUADRATIC_LIST = []
    _INVERSE_LIST = [
        "cometric_matrix",
        "inner_coproduct",
        "hamiltonian",
        "sectional_curvature",
        "scalar_curvature",
    ]
    _INVERSE_SQRT_LIST = ["normalize", "random_unit_tangent_vec", "normal_basis"]
    _RESERVED_NAMES = ("underlying_metric", "scale")

    _SCALING_LISTS = [
        _SQRT_LIST,
        _LINEAR_LIST,
        _QUADRATIC_LIST,
        _INVERSE_LIST,
        _INVERSE_SQRT_LIST,
    ]
    _SCALING_NAMES = ["sqrt", "linear", "quadratic", "inverse", "inverse_sqrt"]

    @classmethod
    def _add_scaled_method(cls, func_name, scaling_type):
        """Configure ScalarProductMetric to scale an attribute.

        This method should be accessed via
        geomstats.geometry.scalar_product_metric.register_scaled_method
        """
        scaling_dict = dict(zip(cls._SCALING_NAMES, cls._SCALING_LISTS))

        for list_of_methods in cls._SCALING_LISTS:
            if func_name in list_of_methods:
                msg = (
                    f"'{func_name}' already has an assigned scaling rule "
                    "which cannot be changed."
                )
                raise ValueError(msg)
        if func_name in cls._RESERVED_NAMES:
            raise ValueError(f"'{func_name}' is reserved for internal use.")

        try:
            scaling_dict[scaling_type].append(func_name)
        except KeyError:
            msg = (
                f"'{scaling_type}' is not an admissible value. Please "
                "provide one of 'sqrt', 'linear', 'quadratic', "
                "'inverse', 'inverse_sqrt'."
            )
            raise ValueError(msg)

    @classmethod
    def _get_scaling_factor(cls, func_name, scale):
        if func_name in cls._SQRT_LIST:
            return gs.sqrt(scale)

        if func_name in cls._LINEAR_LIST:
            return scale

        if func_name in cls._QUADRATIC_LIST:
            return gs.power(scale, 2)

        if func_name in cls._INVERSE_LIST:
            return 1.0 / scale

        if func_name in cls._INVERSE_SQRT_LIST:
            return 1.0 / gs.sqrt(scale)
        return None


class ScalarProductMetric:
    """Class for scalar products of Riemannian and pseudo-Riemannian metrics.

    This class multiplies the (0,2) metric tensor 'space.metric' by a
    scalar 'scaling_factor'. Note that this does not scale distances by
    'scaling_factor'. That would require multiplication by the square of the
    scalar.

    The `space` is not automatically equipped with the `ScalarProductMetric`.

    An object of this type can also be instantiated by the expression
    scaling_factor * space.metric.

    This class acts as a wrapper for the underlying Riemannian metric. All
    public attributes apart from 'underlying_metric' and 'scaling_factor' are
    loaded from the underlying metric at initialization and rescaled by the
    appropriate factor. Changes to the underlying metric at runtime will not
    affect the attributes of this object.

    One exception to this is when the 'underlying_metric' is itself of type
    ScalarProductMetric. In this case, rather than wrapping the wrapper, the
    'underlying_metric' of the first ScalarProductMetric object is wrapped a
    second time with a new 'scaling_factor'.

    Parameters
    ----------
    space : Manifold or ComplexManifold
        A manifold equipped with a metric which is being scaled.
    scale : float
        The value by which to scale the metric. Note that this rescales the
        (0,2) metric tensor, so distances are rescaled by the square root of
        this.
    """

    def __init__(self, space, scale):
        """Load all attributes from the underlying metric."""
        geomstats.errors.check_positive(scale, "scale")
        if not hasattr(space, "metric"):
            raise ValueError("The variable 'space' must be equipped with a metric.")

        self._space = space

        if isinstance(space.metric, ScalarProductMetric):
            self.underlying_metric = space.metric.underlying_metric
            self.scale = scale * space.metric.scale
        else:
            self.underlying_metric = space.metric
            self.scale = scale

        for attr_name in dir(self.underlying_metric):
            if (
                attr_name.startswith("_")
                or attr_name in _ScaledMethodsRegistry._RESERVED_NAMES
            ):
                continue

            attr = getattr(self.underlying_metric, attr_name)
            if not callable(attr):
                try:
                    setattr(self, attr_name, attr)
                except AttributeError as ex:
                    if not isinstance(
                        getattr(type(self.underlying_metric), attr_name, None),
                        property,
                    ):
                        raise ex
            else:
                scale = _ScaledMethodsRegistry._get_scaling_factor(
                    attr_name, self.scale
                )
                method = attr if scale is None else _wrap_attr(scale, attr)
                setattr(self, attr_name, method)

    def __mul__(self, scalar):
        """Multiply the metric by a scalar.

        This method multiplies the (0,2) metric tensor by a scalar. Note that
        this does not scale distances by the scalar. That would require
        multiplication by the square of the scalar.

        Parameters
        ----------
        scalar : float
            The number by which to multiply the metric.

        Returns
        -------
        metric : ScalarProductMetric
            The metric multiplied by the scalar
        """
        if not isinstance(scalar, float):
            return NotImplemented
        return ScalarProductMetric(self._space, scalar)

    def __rmul__(self, scalar):
        """Multiply the metric by a scalar.

        This method multiplies the (0,2) metric tensor by a scalar. Note that
        this does not scale distances by the scalar. That would require
        multiplication by the square of the scalar.

        Parameters
        ----------
        scalar : float
            The number by which to multiply the metric.

        Returns
        -------
        metric : ScalarProductMetric
            The metric multiplied by the scalar.
        """
        return self * scalar
