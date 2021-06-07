"""The Geometry Package."""

from .beta_distributions import BetaDistributions, BetaMetric
from .dirichlet_distributions import DirichletMetric, DirichletDistributions
from .discrete_curves import DiscreteCurves, SRVMetric
from .euclidean import Euclidean, EuclideanMetric
from .full_rank_correlation_matrices import FullRankCorrelationMatrices, \
    FullRankCorrelationAffineQuotientMetric
from .general_linear import GeneralLinear
from .grassmannian import Grassmannian, GrassmannianCanonicalMetric
from .hyperbolic import Hyperbolic
from .hyperboloid import HyperbolicMetric, Hyperboloid, HyperboloidMetric
from .hypersphere import Hypersphere, HypersphereMetric
from .landmarks import Landmarks, L2Metric
from .matrices import Matrices, MatricesMetric
from .minkowski import Minkowski, MinkowskiMetric
from .normal_distributions import NormalDistributions
from .poincare_ball import PoincareBall, PoincareBallMetric
from .poincare_half_space import PoincareHalfSpace, PoincareHalfSpaceMetric
from .pre_shape import KendallShapeMetric, PreShapeSpace, PreShapeMetric
from .product_manifold import ProductManifold, ProductRiemannianMetric
from .skew_symmetric_matrices import SkewSymmetricMatrices
from .spd_matrices import SPDMatrices, SPDMetricAffine, \
    SPDMetricBuresWasserstein, SPDMetricEuclidean, SPDMetricLogEuclidean
from .special_euclidean import SpecialEuclidean, \
    SpecialEuclideanMatrixCannonicalLeftMetric, \
    SpecialEuclideanMatrixLieAlgebra
from .special_orthogonal import SpecialOrthogonal
from .stiefel import Stiefel, StiefelCanonicalMetric
from .symmetric_matrices import SymmetricMatrices

__all__ = [
    'BetaDistributions', 'BetaMetric',
    'DirichletMetric', 'DirichletDistributions',
    'DiscreteCurves', 'SRVMetric',
    'Euclidean', 'EuclideanMetric',
    'FullRankCorrelationMatrices', 'FullRankCorrelationAffineQuotientMetric',
    'GeneralLinear',
    'Grassmannian', 'GrassmannianCanonicalMetric',
    'Hyperboloid', 'HyperboloidMetric',
    'Hyperbolic', 'HyperbolicMetric',
    'Hypersphere', 'HypersphereMetric',
    'Landmarks', 'L2Metric',
    'Matrices', 'MatricesMetric',
    'Minkowski', 'MinkowskiMetric',
    'NormalDistributions',
    'PoincareBall', 'PoincareBallMetric',
    'PoincareHalfSpace', 'PoincareHalfSpaceMetric',
    'ProductManifold', 'ProductRiemannianMetric',
    'KendallShapeMetric', 'PreShapeSpace', 'PreShapeMetric',
    'SkewSymmetricMatrices',
    'SPDMatrices', 'SPDMetricAffine', 'SPDMetricBuresWasserstein',
    'SPDMetricEuclidean', 'SPDMetricLogEuclidean',
    'SpecialEuclidean', 'SpecialEuclideanMatrixCannonicalLeftMetric',
    'SpecialEuclideanMatrixLieAlgebra',
    'SpecialOrthogonal',
    'Stiefel', 'StiefelCanonicalMetric',
    'SymmetricMatrices',
]
