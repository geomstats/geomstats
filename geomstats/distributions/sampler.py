import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.matrices import Matrices, MatricesMetric

class Sampler:
    """Object that can draw a sample from a probability distribution on a manifold"""

    def __init__(self,
                 manifold: Manifold,
                 metric: RiemannianMetric = None):
        self.manifold = manifold
        self.metric = metric if metric else manifold.metric

    def generate_points(self, n_samples):
        samples = self.manifold.random_point(n_samples)
        return samples

