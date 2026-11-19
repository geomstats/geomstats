import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.kmedoids import RiemannianKMedoids


class _FixedInitializationKMedoids(RiemannianKMedoids):
    """Select a known pair while retaining the complete fitting algorithm."""

    def _initialize_medoids(self, distances):
        return gs.copy(self.initial_indices)


@pytest.mark.parametrize("initial_indices", [(4, 5), (5, 4), (3, 5), (5, 3)])
@pytest.mark.parametrize("max_iter", [1, 2, 5])
def test_fitted_kmedoids_labels_match_final_centers(initial_indices, max_iter):
    """An iteration limit must not leave labels for outdated medoids."""
    samples = gs.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    estimator = _FixedInitializationKMedoids(
        Euclidean(1), n_clusters=2, max_iter=max_iter
    )
    estimator.initial_indices = gs.array(initial_indices)

    fitted = estimator.fit(samples)

    centers = estimator.cluster_centers_[:, 0]
    expected = gs.argmin(gs.abs(samples - centers), axis=-1)
    assert fitted is estimator
    np.testing.assert_array_equal(estimator.labels_, expected)
    np.testing.assert_array_equal(estimator.predict(samples), expected)
    np.testing.assert_allclose(
        estimator.cluster_centers_, samples[estimator.medoid_indices_]
    )
