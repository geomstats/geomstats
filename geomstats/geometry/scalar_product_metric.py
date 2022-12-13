"""Scalar product of a Riemannian metric.

Define the product of a Riemannian metric with a scalar number.

Lead author: John Harvey.
"""
from functools import wraps

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric

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
INVERSE_QUADRATIC_LIST = ["normalize", "random_unit_tangent_vec", "normal_basis"]


def _wrap_attr(scaling_factor, func):
    @wraps(func)
    def response(*args, **kwargs):
        res = scaling_factor * func(*args, **kwargs)
        return res

    return response


def _get_scaling_factor(func_name, scale):
    if func_name in SQRT_LIST:
        return gs.sqrt(scale)
    elif func_name in LINEAR_LIST:
        return scale
    elif func_name in QUADRATIC_LIST:
        return gs.power(scale, 2)
    elif func_name in INVERSE_LIST:
        return 1.0 / scale
    elif func_name in INVERSE_QUADRATIC_LIST:
        return 1.0 / gs.power(scale, 2)
    return None


class ScalarProductMetric(RiemannianMetric):
    """Class for scalar products of Riemannian and pseudo-Riemannian metrics.

    This class multiplies the (0,2) metric tensor 'underlying_metric' by a scalar
    'scaling_factor'. Note that this does not scale distances by 'scaling_factor'. That
    would require multiplication by the square of the scalar.

    An object of this type can also be instantiated by the expression
    scaling_factor * underlying_metric.

    This class acts as a wrapper for the underlying Riemannian metric. All public
    attributes apart from 'underlying_metric' and 'scaling_factor' are loaded from the
    underlying metric at initialization and rescaled by the appropriate factor. Changes
    to the underlying metric at runtime will not affect the attributes of this object.

    One exception to this is when the 'underlying_metric' is itself of type
    ScalarProductMetric. In this case, rather than wrapping the wrapper, the
    'underlying_metric' of the first ScalarProductMetric object is wrapped a second
    time with a new 'scaling_factor'.

    Parameters
    ----------
    underlying_metric : RiemannianMetric
        The original metric of the manifold which is being scaled.
    scaling_factor : float
        The value by which to scale the metric. Note that this rescales the (0,2)
        metric tensor, so distances are rescaled by the square root of this.
    """

    def __init__(self, underlying_metric, scaling_factor):
        """Load all attributes from the underlying metric."""
        if hasattr(underlying_metric, "underlying_metric"):
            self.underlying_metric = underlying_metric.underlying_metric
            self.scaling_factor = scaling_factor * underlying_metric.scaling_factor
        else:
            self.underlying_metric = underlying_metric
            self.scaling_factor = scaling_factor

        reserved_names = ("underlying_metric", "scaling_factor")
        for attr_name in dir(self.underlying_metric):
            if attr_name.startswith("_") or attr_name in reserved_names:
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
                scaling_factor = _get_scaling_factor(attr_name, self.scaling_factor)
                method = (
                    attr if scaling_factor is None else _wrap_attr(scaling_factor, attr)
                )
                setattr(self, attr_name, method)
