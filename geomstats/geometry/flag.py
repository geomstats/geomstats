import numpy as np
from geomstats.geometry.manifold import Manifold
import geomstats.backend as gs


class Flag(Manifold):
    def __init__(self, index, n, ambient_manifold):
        d = len(index)
        dim = gs.sum(np.diff(gs.concatenate(([0], index))) * (n-gs.array(index)))  # cf [Ye2021] p 17
        super(Flag, self).__init__(dim=dim, shape=(n*d, n*d), default_point_type="matrix")
        self.n = n
        self.d = d
        self.ambient_manifold = ambient_manifold

    def belongs(self, point, atol=gs.atol):
        pass

    def is_tangent(self, vector, base_point, atol=gs.atol):
        pass

    def to_tangent(self, vector, base_point):
        pass

    def random_point(self, n_samples=1, bound=1.0):
        pass