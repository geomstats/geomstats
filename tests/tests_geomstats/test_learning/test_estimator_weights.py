import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.geometric_median import GeometricMedian
from geomstats.learning.mdm import RiemannianMinimumDistanceToMean


@pytest.mark.parametrize("dtype", [gs.int64, gs.float32, gs.float64])
@pytest.mark.parametrize(
    "estimator_class", [GeometricMedian, RiemannianMinimumDistanceToMean]
)
def test_fit_preserves_weights(estimator_class, dtype):
    """Frequency weights work without changing the caller's array."""
    points = gs.array([[0.0, 1.0], [2.0, 0.0], [4.0, 3.0], [7.0, 6.0]])
    labels = gs.array([0, 0, 1, 1])
    weights = gs.array([1, 2, 3, 4], dtype=dtype)
    original_weights = gs.copy(weights)
    estimator = estimator_class(Euclidean(2))
    reference = estimator_class(Euclidean(2))

    estimator.fit(points, labels, weights=weights)
    reference.fit(points, labels, weights=gs.cast(original_weights, gs.float64))

    assert gs.all(weights == original_weights)
    attribute = "estimate_" if estimator_class is GeometricMedian else "mean_estimates_"
    assert gs.allclose(getattr(estimator, attribute), getattr(reference, attribute))
