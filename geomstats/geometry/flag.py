import numpy as np
import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices


class Flag(Manifold):
    def __init__(self, index, n,
                 ambient_manifold=None):  # set the problem of the structure. List of matrices is not a manifold I guess.
        d = len(index)
        extended_index = gs.concatenate(([0], index), dtype="int")
        dim = int(gs.sum(np.diff(extended_index) * (n - gs.array(index))))  # cf [Ye2021] p 17
        super(Flag, self).__init__(dim=dim, shape=(n * d, n * d),
                                   default_point_type=None)  # I'll implement it as a list of projection matrices
        self.index = index
        self.extended_index = extended_index
        self.d = d
        self.n = n
        self.ambient_manifold = ambient_manifold

    def belongs(self, point, atol=gs.atol):
        # later, must handle several points
        # maybe having a list point of length d+1 like extended_index with the first point being zero would be cleaner?
        # we could do R_i = point[i]. Just, we would have a first point that isn't useful, so memory for nothing
        # and in terms of representation of flags, it wouldn't be rigorous
        belongs = True
        for i in range(1, self.d + 1):
            R_i = point[i - 1]  # the length of point is d while the length of extended indexes is d+1
            belongs = gs.logical_and(belongs, gs.isclose(Matrices.mul(R_i, R_i), R_i, atol=atol).all())
            if not gs.any(belongs):
                return belongs
            belongs = gs.logical_and(belongs, gs.isclose(R_i, Matrices.transpose(R_i), atol=atol).all())
            if not gs.any(belongs):
                return belongs
            belongs = gs.logical_and(belongs, gs.isclose(Matrices.mul(R_i, R_i), Matrices.transpose(R_i), atol=atol).all())
            if not gs.any(belongs):
                return belongs
            belongs = gs.logical_and(belongs, gs.isclose(gs.trace(R_i),
                                                         self.extended_index[i] - self.extended_index[i - 1],
                                                         atol=atol).all())
            if not gs.any(belongs):
                return belongs

            for j in range(1, i):
                R_j = point[j - 1]
                belongs = gs.logical_and(belongs, gs.isclose(Matrices.mul(R_j, R_i), gs.zeros((self.n, self.n)),
                                                             atol=atol).all())
                if not gs.any(belongs):
                    return belongs

        return belongs

    def is_tangent(self, vector, base_point, atol=gs.atol):
        pass

    def to_tangent(self, vector, base_point):
        pass

    def random_point(self, n_samples=1, bound=1.0):
        pass


if __name__ == "__main__":
    flag = Flag([1, 3, 4], 5)
    point1 = [gs.random.rand(5, 5), gs.random.rand(5, 5), gs.random.rand(5, 5)]
    point2 = [gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([0, 1, 1, 0, 0])), gs.array(np.diag([0, 0, 0, 0, 1]))]
    print(flag.belongs(point1))  # False
    print(flag.belongs(point2))  # True
