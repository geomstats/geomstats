import random

import pytest

from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    ElasticMetric,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import (
    BatchGradientDescent,
    FrechetMean,
    GradientDescent,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import BaseEstimatorTestCase
from geomstats.test_cases.learning.frechet_mean import (
    BatchGradientDescentTestCase,
    CircularMeanTestCase,
    ElasticMeanTestCase,
    FrechetMeanTestCase,
    VarianceTestCase,
)

from .data.frechet_mean import (
    BatchGradientDescentTestData,
    CircularMeanTestData,
    FrechetMeanSOCoincideTestData,
    FrechetMeanTestData,
    LinearMeanEuclideaTestData,
    VarianceEuclideanTestData,
    VarianceTestData,
)
