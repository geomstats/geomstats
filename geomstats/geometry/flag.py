import numpy as np
import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices

"""
Implementation of the Flag manifold based on the Grassmannian one.
TODO:
- generalize to several points (returns array instead of single element)
- use gs.all() instead of ().all()
"""


class Flag(Manifold):
    def __init__(self, index, n, ambient_manifold=None):
        # set the problem of the structure. List of matrices is not a manifold I guess.
        # using block diagonal matrices like in the paper would be cool because of the SPD structure,
        # but too memory expensive
        d = len(index)
        geomstats.errors.check_integer(d, "d")
        geomstats.errors.check_integer(n, "n")
        extended_index = gs.concatenate(([0], index), dtype="int")
        dim = int(gs.sum(np.diff(extended_index) * (n - gs.array(index))))  # cf [Ye2021] p 17
        super(Flag, self).__init__(dim=dim, shape=(n * d, n * d),
                                   default_point_type=None)  # I'll implement it as a list of projection matrices
        self.index = index
        self.extended_index = extended_index
        self.d = d
        self.n = n
        self.ambient_manifold = ambient_manifold

    def belongs(self, point, atol=gs.atol):  # characterization from [Ye2021] Proposition 21
        belongs = True  # just to initialize, be actually we will anyway go into the loop as d is supposed > 0
        for i in range(1, self.d + 1):
            R_i = point[i - 1]  # the length of point is d while the length of extended indexes is d+1
            eq1 = gs.isclose(Matrices.mul(R_i, R_i), R_i, atol=atol).all()
            eq2 = gs.isclose(R_i, Matrices.transpose(R_i), atol=atol).all()
            eq3 = gs.isclose(Matrices.mul(R_i, R_i), Matrices.transpose(R_i), atol=atol).all()
            eq4 = gs.isclose(gs.trace(R_i), self.extended_index[i] - self.extended_index[i - 1], atol=atol).all()
            belongs = gs.all([eq1, eq2, eq3, eq4])

            for j in range(1, i):
                R_j = point[j - 1]
                belongs = gs.logical_and(belongs, gs.isclose(Matrices.mul(R_j, R_i), gs.zeros((self.n, self.n)),
                                                             atol=atol).all())
            if not gs.any(belongs):
                return belongs

        return belongs

    def is_tangent(self, vector, base_point, atol=gs.atol):  # characterization from [Ye2021] Proposition 22
        is_tangent = True  # just to initialize, be actually we will anyway go into the loop as d is supposed > 0
        for i in range(1, self.d + 1):
            R_i = base_point[i - 1]  # the length of point is d while the length of extended indexes is d+1
            Z_i = vector[i - 1]
            eq1 = gs.isclose(Matrices.mul(R_i, Z_i) + Matrices.mul(Z_i, R_i), Z_i, atol=atol).all()
            eq2 = gs.isclose(Z_i, Matrices.transpose(Z_i), atol=atol).all()
            eq3 = gs.isclose(Matrices.mul(R_i, Z_i) + Matrices.mul(Z_i, R_i), Matrices.transpose(Z_i), atol=atol).all()
            eq4 = gs.isclose(gs.trace(Z_i), gs.zeros((self.n, self.n)), atol=atol).all()
            is_tangent = gs.all([eq1, eq2, eq3, eq4])

            for j in range(1, i):
                R_j = base_point[j - 1]
                Z_j = vector[j - 1]
                is_tangent = gs.logical_and(is_tangent, gs.isclose(Matrices.mul(Z_i, R_j) + Matrices.mul(R_i, Z_j),
                                                                   gs.zeros((self.n, self.n)),
                                                                   atol=atol).all())
            if not gs.any(is_tangent):
                return is_tangent

        return is_tangent

    def to_tangent(self, vector, base_point):
        pass

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points from a uniform distribution.
        We use the same trick as for Grassmannian class for the random uniform projection matrix sampling and iterate
        over the reduced projection coordinates with the constraint that the randomly generated projection matrix needs
        to be in the orthogonal of the previous indexes' vector spaces (cf [Ye2021] Proposition 21).

        Parameters
        ----------
        n_samples : int
            The number of points to sample
            Optional. default: 1.

        Returns
        -------
        projectors : array-like, shape=[..., n, n]
            Points following a uniform distribution.

        References
        ----------
        .. [Chikuse03] Yasuko Chikuse, Statistics on special manifolds,
        New York: Springer-Verlag. 2003, 10.1007/978-0-387-21540-2
        .. [Ye2021] Ye, K., Wong, K.S.-W., Lim, L.-H.: Optimization on flag manifolds.
        Preprint, arXiv:1907.00949 [math.OC] (2019)
        """
        for i in range(1, self.d+1):
            dim_proj = self.extended_index[i] - self.extended_index[i - 1]
            points = gs.random.normal(size=(n_samples, self.n, dim_proj))
            full_rank = Matrices.mul(Matrices.transpose(points), points)
            projector = Matrices.mul(
                points, GeneralLinear.inverse(full_rank), Matrices.transpose(points)
            )
            # randomly sample on the orthogonal characterized by the matrix P (1-D) P.T, with dim = dim_proj
            # does randomly sampling then orthogonally projecting with I - P work?
            # Nope I don't think so... There would be way too much zeroes...
        return projector[0] if n_samples == 1 else projector


if __name__ == "__main__":
    # flag = Flag([1, 3, 4], 5)
    # point1 = [gs.random.rand(5, 5), gs.random.rand(5, 5), gs.random.rand(5, 5)]
    # point2 = [gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([0, 1, 1, 0, 0])),
    #           gs.array(np.diag([0, 0, 0, 0, 1]))]
    # print(flag.belongs(point1))  # False
    # print(flag.belongs(point2))  # True
    # print(flag.is_tangent(point2, base_point=point1))  # False
    # print(flag.is_tangent(point1, base_point=point2))  # False
    #
    from geomstats.geometry.grassmannian import Grassmannian
    from functools import reduce

    grassmannian = Grassmannian(10, 2)
    proj1 = grassmannian.random_point()
    proj1_perp = gs.eye(10) - proj1
    points = gs.random.normal(size=(1000, 10, 2))  # Trace is always 2, even for 100,000 samples
    points_perp = Matrices.mul(proj1_perp, points)
    full_rank_perp = Matrices.mul(Matrices.transpose(points_perp), points_perp)
    proj2 = Matrices.mul(
        points_perp, GeneralLinear.inverse(full_rank_perp), Matrices.transpose(points_perp)
    )
    print((gs.all([gs.isclose(gs.trace(p), 2) for p in proj2])))
    print((gs.all([gs.isclose(Matrices.mul(proj1, p), gs.zeros((10, 10))) for p in proj2])))