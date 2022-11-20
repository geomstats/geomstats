"""Scalar product of a Riemannian metric.

Define the product of a Riemannian metric with a scalar number.

Lead author: John Harvey.
"""
from functools import wraps

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric


INVARIANT_LIST = [
    "christoffels", "geodesic_equation", "exp", "log", "_pole_ladder_step",
    "_schild_ladder_step", "ladder_parallel_transport", "riemann_tensor", "curvature",
    "ricci_tensor", "directional_curvature", "curvature_derivative",
    "directional_curvature_derivative", "geodesic", "parallel_transport",
    "closest_neighbor_index"]
SQRT_LIST = ["norm", "dist", "dist_broadcast", "dist_pairwise", "diameter"]
LINEAR_LIST = [
    "metric_matrix", "inner_product", "inner_product_derivative_matrix",
    "squared_norm", "squared_dist", "covariant_riemann_tensor"]
QUADRATIC_LIST = []
INVERSE_LIST = [
    "cometric_matrix", "inner_coproduct", "hamiltonian", "sectional_curvature",
    "scalar_curvature"]
INVERSE_QUADRATIC_LIST = ["normalize", "random_unit_tangent_vec", "normal_basis"]


def _wrap_attr(scale, underlying_metric, func):
    @wraps(func)
    def response(*args, **kwargs):
        if func.__name__ in INVARIANT_LIST:
            scaling_factor = 1
        elif func.__name__ in SQRT_LIST:
            scaling_factor = gs.sqrt(scale)
        elif func.__name__ in LINEAR_LIST:
            scaling_factor = scale
        elif func.__name__ in QUADRATIC_LIST:
            scaling_factor = gs.power(scale, 2)
        elif func.__name__ in INVERSE_LIST:
            scaling_factor = 1.0 / scale
        elif func.__name__ in INVERSE_QUADRATIC_LIST:
            scaling_factor = 1.0 / gs.power(scale, 2)
        else:
            scaling_factor = None

        if scaling_factor is None:
            raise AttributeError(
                f"Object of class 'ScalarProductMetric' cannot transform attribute "
                f"'{func.__name__}' of the underlying metric.")
        res = scaling_factor * func(*args, **kwargs)
        return res

    return response


class ScalarProductMetricV2(RiemannianMetric):
    """Class for scalar products of Riemannian and pseudo-Riemannian metrics.

    This class acts as a wrapper for the underlying Riemannian metric. All attributes
    apart from 'underlying_metric' and 'scale' are loaded from the underlying metric as
    needed and so changes to the underlying metric at runtime will change this object.

    Parameters
    ----------
    underlying_metric : RiemannianMetric
        The original metric of the manifold which is being scaled.
    scaling_factor : float
        The value by which to scale the metric. Note that this rescales the metric, so
        distances are rescaled by the square root of this.
    """

    def __init__(self, underlying_metric, scaling_factor):
        self.underlying_metric = underlying_metric
        self.scaling_factor = scaling_factor

    def __getattribute__(self, attr):
        """Load all attributes from the underlying metric as needed."""
        if attr in ['scaling_factor', 'underlying_metric', '__dict__']:
            return object.__getattribute__(self, attr)
        val = getattr(self.underlying_metric, attr)
        if not callable(val):
            return val
        else:
            return _wrap_attr(self.scaling_factor, self.underlying_metric, val)
