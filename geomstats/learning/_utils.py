from geomstats.geometry.discrete_curves import ElasticMetric, SRVMetric
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.metric_geometry.graph_space import GraphSpace
from geomstats.metric_geometry.point_set import PointSetMetric

ELASTIC_METRICS = (SRVMetric, ElasticMetric)


def _is_linear_metric(metric):
    return isinstance(metric, EuclideanMetric)


def _is_elastic_metric(metric):
    return isinstance(metric, tuple(ELASTIC_METRICS))


def _is_geodesic_metric(metric):
    return isinstance(metric, PointSetMetric)


def _is_graph_space(space):
    return isinstance(space, GraphSpace)
