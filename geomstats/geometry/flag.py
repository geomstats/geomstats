r"""
Manifold of flags.

The flag manifold :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)` is a smooth manifold whose elements are
flags in a vector space of dimension n, i.e. nested sequences of linear subspaces with increasing
dimensions :math:`n_0:=0 < n_1 < n_2 < \dots < n_d < n_{d+1}:=n`.

Lead author: Tom Szwagier.

:math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)` is represented by
:math:`nd \times nd` block diagonal matrices, where each block :math:`i \in \{1, \dots, d\}` corresponds to a
:math:`n \times n` matrix :math:`R_i` of rank :math:`n_i-n_{i-1}` satisfying :math:`{R_i}^2 = R_i = {R_i}^\top` and
:math:`R_i R_j = 0` for j < i. The mapping is diffeomorphic (cf. [Ye2021] Proposition 21).
Each :math:`R_i \in \operatorname{Flag}(n_1, n_2 \dots, n_d; n)` is thus identified with the unique orthogonal projector
onto :math:`{\rm Im}(R_i)`, with the constraint that the related subspaces must be orthogonal one to another.

:math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)` can also be seen as a matrix homogeneous space:

.. math::

    \operatorname{Flag}(n_1, n_2 \dots, n_d; n) \simeq \frac {O(n)} {O(n_1) \times O(n_2 - n_1) \times \dots \times
    O(n-n_d)}

References
----------
.. [Ye2021] Ye, K., Wong, K.S.-W., Lim, L.-H.: Optimization on flag manifolds.
Preprint, arXiv:1907.00949 [math.OC] (2019)
"""

import numpy as np
import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices


class Flag(Manifold):
    """ Class for flag manifolds :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)`.

    Parameters
    ----------
    index : tuple of int
        Sequence of dimensions of the nested linear subspaces.
    n : int
        Dimension of the Euclidean space.
    """

    def __init__(self, index, n):
        # set the problem of the structure. List of matrices is not a manifold I guess.
        # using block diagonal matrices like in the paper would be cool because of the SPD structure,
        # but too memory expensive
        d = len(index)
        geomstats.errors.check_integer(d, "d")
        geomstats.errors.check_integer(n, "n")
        extended_index = gs.concatenate(([0], index), dtype="int")
        dim = int(gs.sum(np.diff(extended_index) * (n - gs.array(index))))  # cf [Ye2021] p 17
        super(Flag, self).__init__(dim=dim, shape=(n * d, n * d))
        self.index = index
        self.extended_index = extended_index
        self.d = d
        self.n = n

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Characterization based on reduced projection coordinates from [Ye2021], Proposition 21:
        **Proposition 21**
        The flag manifold :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)` is diffeomorphic to
        .. math::
            \left\{\R = \operatorname{diag}\left(R_1, \dots, R_d\right) \in \mathbb{R}^{nd \times nd} :
            {R_i}^2 = R_i = {R_i}^\top, \operatorname[{tr}(R_i)=n_i-n_{i-1}, R_i R_j = 0, j < i right\}


        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        def each_belongs(pt):
            for i in range(1, self.d + 1):
                R_i = pt[i - 1]
                eq1 = gs.all(gs.isclose(Matrices.mul(R_i, R_i), R_i, atol=atol))
                eq2 = gs.all(gs.isclose(R_i, Matrices.transpose(R_i), atol=atol))
                eq3 = gs.all(gs.isclose(Matrices.mul(R_i, R_i), Matrices.transpose(R_i), atol=atol))
                eq4 = gs.isclose(gs.trace(R_i),
                                 self.extended_index[i] - self.extended_index[i - 1], atol=atol)
                belongs = gs.all([eq1, eq2, eq3, eq4])
                if not gs.any(belongs):
                    return belongs

                for j in range(1, i):
                    R_j = pt[j - 1]
                    belongs = gs.all(gs.isclose(Matrices.mul(R_j, R_i), gs.zeros((self.n, self.n)), atol=atol))
                    if not gs.any(belongs):
                        return belongs
            return belongs

        if isinstance(point, list) or point.ndim > 3:
            return gs.stack([each_belongs(pt) for pt in point])

        return each_belongs(point)

    def is_tangent(self, vector, base_point, atol=gs.atol):  # characterization from [Ye2021] Proposition 22

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
        pass


if __name__ == "__main__":
    flag = Flag([1, 3, 4], 5)
    point1 = gs.random.rand(100, 3, 5, 5)
    point2 = gs.array([gs.array(np.diag([1, 0, 0, 0, 0])), gs.array(np.diag([0, 1, 1, 0, 0])),
                       gs.array(np.diag([0, 0, 0, 0, 1]))])
    print(flag.belongs(point1))  # False
    print(flag.belongs(point2))  # True
    print(flag.is_tangent(point2, base_point=point1))  # False
    print(flag.is_tangent(point1, base_point=point2))  # False

    # from geomstats.geometry.grassmannian import Grassmannian
    # from functools import reduce
    # grassmannian = Grassmannian(10, 2)
    # p1 = grassmannian.random_point()
    # p2 = grassmannian.random_point(2)
    # b1 = grassmannian.belongs(p1)
    # b2 = grassmannian.belongs(p2)

    # proj1 = grassmannian.random_point()
    # proj1_perp = gs.eye(10) - proj1
    # points = gs.random.normal(size=(1000, 10, 2))  # Trace is always 2, even for 100,000 samples
    # points_perp = Matrices.mul(proj1_perp, points)
    # full_rank_perp = Matrices.mul(Matrices.transpose(points_perp), points_perp)
    # proj2 = Matrices.mul(
    #     points_perp, GeneralLinear.inverse(full_rank_perp), Matrices.transpose(points_perp)
    # )
    # print((gs.all([gs.isclose(gs.trace(p), 2) for p in proj2])))
    # print((gs.all([gs.isclose(Matrices.mul(proj1, p), gs.zeros((10, 10))) for p in proj2])))
