"""The Geometry Package."""
from .base import LevelSet, OpenSet, VectorSpace
from .connection import Connection
from .discrete_curves import (
    ClosedDiscreteCurves,
    ClosedSRVMetric,
    DiscreteCurves,
    QuotientSRVMetric,
    SRVMetric,
)
from .euclidean import Euclidean, EuclideanMetric
from .fiber_bundle import FiberBundle
from .full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationAffineQuotientMetric,
    FullRankCorrelationMatrices,
)
from .general_linear import GeneralLinear
from .grassmanian import Grassmannian, GrassmannianCanonicalMetric
from .matrices import Matrices
