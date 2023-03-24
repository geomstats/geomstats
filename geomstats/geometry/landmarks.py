"""Manifold for sets of landmarks that belong to any given manifold.

Lead author: Nicolas Guigui.
"""

from geomstats.geometry.nfold_manifold import NFoldManifold, NFoldMetric


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

    def __init__(self, ambient_manifold, k_landmarks, equip=True):
        self.k_landmarks = k_landmarks
        super().__init__(
            base_manifold=ambient_manifold, n_copies=k_landmarks, equip=equip
        )

    def _default_metric(self):
        return L2LandmarksMetric


class L2LandmarksMetric(NFoldMetric):
    """L2 Riemannian metric on the space of landmarks.

    This is the NFoldMetric of the n-fold manifold made out
    of k_landmarks copies of the ambient manifold of each landmark.
    """