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
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices


class Flag(Manifold):
    """ Class for flag manifolds :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)`.
    Representation, notations and formulas inspired from [Ye2021].

    References
    ----------
    .. [Ye2021] Ye, K., Wong, K.S.-W., Lim, L.-H.: Optimization on flag manifolds.
    Preprint, arXiv:1907.00949 [math.OC] (2019)

    Parameters
    ----------
    index : tuple of int
        Sequence of dimensions of the nested linear subspaces.
    n : int
        Dimension of the Euclidean space.
    """

    def __init__(self, index, n):
        d = len(index)
        geomstats.errors.check_integer(d, "d")
        geomstats.errors.check_integer(n, "n")
        extended_index = gs.concatenate(([0], index), dtype="int")
        dim = int(gs.sum(np.diff(extended_index) * (n - gs.array(index))))
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
            {R_i}^2 = R_i = {R_i}^\top, \operatorname[{tr}(R_i)=n_i-n_{i-1}, R_i R_j = 0, i < j right\}

        References
        ----------
        .. [Ye2021] Ye, K., Wong, K.S.-W., Lim, L.-H.: Optimization on flag manifolds.
        Preprint, arXiv:1907.00949 [math.OC] (2019)


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
                if not belongs:
                    return belongs

                for j in range(1, i):
                    R_j = pt[j - 1]
                    belongs = gs.all(gs.isclose(Matrices.mul(R_j, R_i), gs.zeros((self.n, self.n)), atol=atol))
                    if not belongs:
                        return belongs

            return belongs

        if isinstance(point, list) or point.ndim > 3:
            return gs.stack([each_belongs(pt) for pt in point])

        return each_belongs(point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Characterization based on reduced projection coordinates from [Ye2021], Proposition 22:
        **Proposition 22**
        Let :math:`\R = \operatorname{diag}\left(R_1, \dots, R_d\right) \in
        \operatorname{Flag}(n_1, n_2 \dots, n_d; n)`.
         Then the tangent space is given by

        .. math::
            T_R \operatorname{Flag}(n_1, n_2 \dots, n_d; n) =
            \left\{Z = \operatorname{diag}\left(Z_1, \dots, Z_d\right) \in \mathbb{R}^{nd \times nd} :
            R_i Z_i + Z_i R_i = Z_i = {Z_i}^\top, \operatorname[{tr}(Z_i)=0, Z_i R_j + R_i Z_j= 0, i < j right\}

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """

        def each_is_tangent(vec, bp):
            for i in range(1, self.d + 1):
                R_i = bp[i - 1]
                Z_i = vec[i - 1]
                eq1 = gs.all(gs.isclose(Matrices.mul(R_i, Z_i) + Matrices.mul(Z_i, R_i), Z_i, atol=atol))
                eq2 = gs.all(gs.isclose(Z_i, Matrices.transpose(Z_i), atol=atol))
                eq3 = gs.all(gs.isclose(Matrices.mul(R_i, Z_i) + Matrices.mul(Z_i, R_i), Matrices.transpose(Z_i),
                                        atol=atol))
                eq4 = gs.isclose(gs.trace(Z_i), 0, atol=atol)
                is_tangent = gs.all([eq1, eq2, eq3, eq4])
                if not is_tangent:
                    return is_tangent

                for j in range(1, i):
                    R_j = bp[j - 1]
                    Z_j = vec[j - 1]
                    is_tangent = gs.all(gs.isclose(Matrices.mul(Z_i, R_j) + Matrices.mul(R_i, Z_j),
                                                   gs.zeros((self.n, self.n)),
                                                   atol=atol))
                    if not is_tangent:
                        return is_tangent
            return is_tangent

        if isinstance(base_point, list) or base_point.ndim > 3:
            return gs.stack([each_is_tangent(vec, bp) for (vec, bp) in zip(vector, base_point)])

        return each_is_tangent(vector, base_point)

    def to_tangent(self, vector, base_point):
        pass

    def random_point(self, n_samples=1, bound=1.0):
        pass
