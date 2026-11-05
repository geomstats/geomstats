import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.kmeans import RiemannianKMeans


@pytest.mark.parametrize("n_clusters", [1, 2])
@pytest.mark.parametrize("n_queries", [1, 4])
def test_prediction_keeps_sample_and_cluster_axes(n_clusters, n_queries):
    """A singleton sample or cluster still produces one label per sample."""
    points = gs.array([[-2.0], [-1.0], [1.0], [2.0]])
    initial = gs.array([[-2.0], [2.0]])[:n_clusters]
    estimator = RiemannianKMeans(Euclidean(1), n_clusters=n_clusters, init=initial).fit(
        points
    )

    predictions = estimator.predict(points[:n_queries])

    expected = gs.array([0, 0, 1, 1] if n_clusters == 2 else [0, 0, 0, 0])
    assert predictions.shape == (n_queries,)
    assert gs.all(predictions == expected[:n_queries])
