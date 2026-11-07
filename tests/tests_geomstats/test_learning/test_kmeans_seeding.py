import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning import kmeans


def test_kmeans_plus_plus_weights_squared_distances(monkeypatch):
    captured = []

    class Distribution:
        def __init__(self, values):
            captured.extend(values[1])

        def rvs(self):
            return 2

    monkeypatch.setattr(kmeans, "randint", lambda *_: 0)
    monkeypatch.setattr(kmeans, "rv_discrete", Distribution)
    model = kmeans.RiemannianKMeans(Euclidean(1), 2, init="kmeans++")
    model._pick_init_cluster_centers(gs.array([[0.0], [1.0], [3.0]]))
    assert captured == pytest.approx([0.0, 0.1, 0.9])


def test_kmeans_plus_plus_identical_points():
    points = gs.ones((3, 2))
    model = kmeans.RiemannianKMeans(Euclidean(2), 3, init="kmeans++").fit(points)
    assert gs.allclose(model.cluster_centers_, points)


def test_random_initialization_selects_distinct_samples():
    points = gs.reshape(gs.arange(10), (10, 1))
    model = kmeans.RiemannianKMeans(Euclidean(1), 10)
    assert gs.all(gs.sort(model._pick_init_cluster_centers(points), axis=0) == points)


@pytest.mark.parametrize("callable_init", [False, True])
def test_fit_preserves_supplied_centers(callable_init):
    centers = gs.array([[0.0]])
    init = (lambda *_: centers) if callable_init else centers
    model = kmeans.RiemannianKMeans(Euclidean(1), 1, init=init)
    model.fit(gs.array([[2.0], [4.0]]))
    assert gs.allclose(centers, gs.array([[0.0]]))
    assert gs.allclose(model.cluster_centers_, gs.array([[3.0]]))


def test_validate_all_point_dimensions():
    model = kmeans.RiemannianKMeans(Euclidean(2), 2, init=gs.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="Dimensions"):
        model._pick_init_cluster_centers(gs.zeros((4, 2, 2)))


@pytest.mark.parametrize("n_clusters", [0, 3])
def test_fit_rejects_invalid_cluster_count(n_clusters):
    model = kmeans.RiemannianKMeans(Euclidean(1), n_clusters)
    with pytest.raises(ValueError, match="n_clusters must be between 1 and"):
        model.fit(gs.array([[0.0], [1.0]]))
