"""Scalar product of a Riemannian metric.

Define the product of a Riemannian metric with a scalar number.

Lead author: John Harvey.
"""

from functools import wraps

import geomstats.backend as gs
import geomstats.errors


def _wrap_attr(scaling_factor, func):
    @wraps(func)
    def response(*args, **kwargs):
        res = scaling_factor * func(*args, **kwargs)
        return res

    return response


class ScalarProductMetric:
    """Class for scalar products of Riemannian and pseudo-Riemannian metrics.

    This class multiplies the (0,2) metric tensor 'underlying_metric' by a
    scalar 'scaling_factor'. Note that this does not scale distances by
    'scaling_factor'. That would require multiplication by the square of the
    scalar.

    An object of this type can also be instantiated by the expression
    scaling_factor * underlying_metric.

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
    underlying_metric : RiemannianMetric
        The original metric of the manifold which is being scaled.
    scale : float
        The value by which to scale the metric. Note that this rescales the
        (0,2) metric tensor, so distances are rescaled by the square root of
        this.
    """

    SQRT_LIST = ["norm", "dist", "dist_broadcast", "dist_pairwise", "diameter"]
    LINEAR_LIST = [
        "metric_matrix",
        "inner_product",
        "inner_product_derivative_matrix",
        "squared_norm",
        "squared_dist",
        "covariant_riemann_tensor",
    ]
    QUADRATIC_LIST = []
    INVERSE_LIST = [
        "cometric_matrix",
        "inner_coproduct",
        "hamiltonian",
        "sectional_curvature",
        "scalar_curvature",
    ]
    INVERSE_SQRT_LIST = ["normalize", "random_unit_tangent_vec", "normal_basis"]
    RESERVED_NAMES = ("underlying_metric", "scale")

    def __init__(self, underlying_metric, scale):
        """Load all attributes from the underlying metric."""
        geomstats.errors.check_positive(scale, "scale")

        if hasattr(underlying_metric, "underlying_metric"):
            self.underlying_metric = underlying_metric.underlying_metric
            self.scale = scale * underlying_metric.scale
        else:
            self.underlying_metric = underlying_metric
            self.scale = scale

        for attr_name in dir(self.underlying_metric):
            if attr_name.startswith("_") or attr_name in type(self).RESERVED_NAMES:
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
                scale = type(self)._get_scaling_factor(attr_name, self.scale)
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
        return ScalarProductMetric(self, scalar)

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

    @classmethod
    def _get_scaling_factor(cls, func_name, scale):
        if func_name in cls.SQRT_LIST:
            return gs.sqrt(scale)

        if func_name in cls.LINEAR_LIST:
            return scale

        if func_name in cls.QUADRATIC_LIST:
            return gs.power(scale, 2)

        if func_name in cls.INVERSE_LIST:
            return 1.0 / scale

        if func_name in cls.INVERSE_SQRT_LIST:
            return 1.0 / gs.sqrt(scale)
        return None

    @classmethod
    def add_scaled_method(cls, func_name, scaling_type):
        """Configure ScalarProductMetric to scale an attribute.

        The ScalarProductMetric class rescales various methods of a
        RiemannianMetric by the correct factor. The default behaviour is to
        rescale linearly. This method allows the use to add a new method to be
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
        scaling_lists = [
            cls.SQRT_LIST,
            cls.LINEAR_LIST,
            cls.QUADRATIC_LIST,
            cls.INVERSE_LIST,
            cls.INVERSE_SQRT_LIST,
        ]
        scaling_types = ["sqrt", "linear", "quadratic", "inverse", "inverse_sqrt"]
        scaling_dict = dict(zip(scaling_types, scaling_lists))

        for list_of_methods in scaling_lists:
            if func_name in list_of_methods:
                msg = (
                    f"'{func_name}' already has an assigned scaling rule "
                    "which cannot be changed."
                )
                raise ValueError(msg)
        if func_name in cls.RESERVED_NAMES:
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
