"""Manifold for sets of landmarks that belong to any given manifold.

Lead author: Nicolas Guigui.
"""

from geomstats.geometry.product_manifold import NFoldManifold
from geomstats.geometry.product_riemannian_metric import NFoldMetric


class Landmarks(NFoldManifold):
    """Class for space of landmarks.

    The landmark space is a product manifold where all manifolds in the
    product are the same. The default metric is the product metric and
    is often referred to as the L2 metric.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold to which landmarks belong.
    k_landmarks : int
        Number of landmarks.
    """

    def __init__(self, ambient_manifold, k_landmarks, **kwargs):
        kwargs.setdefault(
            "metric", L2LandmarksMetric(ambient_manifold.metric, k_landmarks)
        )
        super().__init__(base_manifold=ambient_manifold, n_copies=k_landmarks, **kwargs)
        self.ambient_manifold = ambient_manifold
        self.k_landmarks = k_landmarks


class L2LandmarksMetric(NFoldMetric):
    """L2 Riemannian metric on the space of landmarks.

    This is the NFoldMetric of the n-fold manifold made out
    of k_landmarks copies of the ambient manifold of each landmark.

    Parameters
    ----------
    ambient_metric : RiemannianMetric
        Riemannian metric of the manifold to which the landmarks belong.
    k_landmarks: int
        Number of landmarks.

    """

    def __init__(self, ambient_metric, k_landmarks, **kwargs):
        super().__init__(base_metric=ambient_metric, n_copies=k_landmarks, **kwargs)
        self.ambient_metric = ambient_metric
        self.k_landmarks = k_landmarks
