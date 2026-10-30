import logging

from geomstats.geometry.discrete_curves import ElasticMetric, SRVMetric
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.metric_geometry.graph_space import GraphSpace
from geomstats.metric_geometry.point_set import PointSetMetric

logger = logging.getLogger(__name__)

ELASTIC_METRICS = (SRVMetric, ElasticMetric)


def _warn_max_iterations(max_iter):
    """Warn that the maximum number of iterations was reached.

    Callers decide when the iteration budget is exhausted.
    """
    logger.warning(
        f"Maximum number of iterations {max_iter} reached. "
        "The estimate may be inaccurate."
    )


def _is_linear_metric(metric):
    return isinstance(metric, EuclideanMetric)


def _is_elastic_metric(metric):
    return isinstance(metric, tuple(ELASTIC_METRICS))


def _is_geodesic_metric(metric):
    return isinstance(metric, PointSetMetric)


def _is_graph_space(space):
    return isinstance(space, GraphSpace)
