"""The flag manifold (Work In Progress).

Lead author: Tom Szwagier.
"""
import os
from scipy.linalg import polar

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OMP Error with Pytorch backend.


# CAUTION may cause crashes or silently produce incorrect results. cf stackoverflow


class Flag(Manifold):
    r"""Class for flag manifolds.

    Representation, notations and formulas inspired from **[Ye2021]**.

    The flag manifold :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)` is a smooth
    manifold whose elements are flags in a vector space of dimension n, i.e. nested
    sequences of linear subspaces with increasing dimensions :math:`n_0:=0 < n_1 <
    n_2 < \dots < n_d < n_{d+1}:=n`.

    :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)` is represented by
    :math:`nd \times nd` block diagonal matrices, where each block
    :math:`i \in \{1, \dots, d\}` corresponds to a
    :math:`n \times n` matrix :math:`R_i` of rank :math:`n_i-n_{i-1}`
    satisfying :math:`{R_i}^2 = R_i = {R_i}^\top` and
    :math:`R_i R_j = 0` for j < i. The mapping is diffeomorphic
    (cf. **[Ye2021]** Proposition 21).
    Each :math:`R_i \in \operatorname{Flag}(n_1, n_2 \dots, n_d; n)` is thus identified
    with the unique orthogonal projector onto :math:`{\rm Im}(R_i)`,
    with the constraint that the related subspaces must be orthogonal one to another.

    :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)` can also be seen as a matrix
    homogeneous space:

    .. math::

        \operatorname{Flag}(n_1, n_2 \dots, n_d; n) \simeq \frac {O(n)} {O(n_1)
        \times O(n_2 - n_1) \times \dots \times
        O(n-n_d)}

    Contrarily to what is done in **[Ye2021]**, for memory reasons, we here represent
    the points as sequences of $d$ $n \times n$ projection matrices, as the block
    diagonal representation doesn't add anything from the computational point of view.
    Hence the `shape` of a point is (d, n, n). That representation is subject to
    evolve through time.

    This implementation is part of an open research project (master thesis + PhD),
    so the Pull Request will come only once the representation is set,
    the methods are proven to work, and we get some results that can be illustrated
    through a Notebook.

    References
    ----------
    .. **[Ye2021]** Ye, K., Wong, K.S.-W., Lim, L.-H.: Optimization on flag manifolds.
    Preprint, arXiv:1907.00949 [math.OC] (2019)

    Parameters
    ----------
    index : list of int
        Sequence of dimensions of the nested linear subspaces.
    n : int
        Dimension of the Euclidean space.
    """

    def __init__(self, n, index):
        index = gs.array(index)
        d = len(index)
        geomstats.errors.check_integer(d, "d")
        geomstats.errors.check_integer(n, "n")
        extended_index = gs.concatenate((gs.array([0]), index))
        dim = int(gs.sum((extended_index[1:] - extended_index[:-1]) * (n - index)))
        super(Flag, self).__init__(
            dim=dim, shape=(d, n, n), default_point_type="matrix"
        )  # that's not true...
        # should I change the representation to diag per block matrix (or Stiefel
        # like, but risk of non-unicity) ? We'll see later, if I don't have errors
        # for now
        self.n = n
        self.d = d
        self.index = index
        self.extended_index = extended_index

    def belongs(self, point, atol=gs.atol):
        r"""Evaluate if a point belongs to the manifold.

        Characterization based on reduced projection coordinates from **[Ye2021]**.

        **Proposition 21:**
        The flag manifold :math:`\operatorname{Flag}(n_1, n_2 \dots, n_d; n)`
        is diffeomorphic to:

        .. math::
            \left\{R = \operatorname{diag}\left(R_1, \dots, R_d\right)
            \in \mathbb{R}^{nd \times nd} :
            {R_i}^2 = R_i = {R_i}^\top,
            \operatorname{tr}(R_i)=n_i-n_{i-1}, R_i R_j = 0, i < j \right\}
        |
        Parameters
        ----------
        point : array-like, shape=[..., d, n, n]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """

        def _each_belongs(pt):
            """Auxiliary function to deal with samples one at a time."""
            for i in range(1, self.d + 1):
                R_i = pt[i - 1]
                cst_1 = gs.all(gs.isclose(Matrices.mul(R_i, R_i), R_i, atol=atol))
                cst_2 = gs.all(gs.isclose(R_i, Matrices.transpose(R_i), atol=atol))
                cst_3 = gs.all(
                    gs.isclose(
                        Matrices.mul(R_i, R_i), Matrices.transpose(R_i), atol=atol
                    )
                )
                cst_4 = gs.isclose(
                    gs.trace(R_i),
                    float(self.extended_index[i] - self.extended_index[i - 1]),
                    atol=atol,
                )
                belongs = gs.all([cst_1, cst_2, cst_3, cst_4])
                if not belongs:
                    return belongs

                for j in range(1, i):
                    R_j = pt[j - 1]
                    belongs = gs.all(
                        gs.isclose(
                            Matrices.mul(R_j, R_i),
                            gs.zeros((self.n, self.n)),
                            atol=atol,
                        )
                    )
                    if not belongs:
                        return belongs

            return belongs

        if isinstance(point, list) or point.ndim > 3:
            return gs.stack([_each_belongs(pt) for pt in point])

        return _each_belongs(point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        r"""Check whether the vector is tangent at base_point.

        Characterization based on reduced projection coordinates from **[Ye2021]**.

        **Proposition 22:**
        Let :math:`R = \operatorname{diag}\left(R_1, \dots, R_d\right) \in
        \operatorname{Flag}(n_1, n_2 \dots, n_d; n)`.
        Then the tangent space at point R is given by:

        .. math::
            T_R \operatorname{Flag}(n_1, n_2 \dots, n_d; n) =
            \left\{Z = \operatorname{diag}\left(Z_1, \dots, Z_d\right)
            \in \mathbb{R}^{nd \times nd} :
            R_i Z_i + Z_i R_i = Z_i = {Z_i}^\top,
            \operatorname{tr}(Z_i)=0, Z_i R_j + R_i Z_j= 0, i < j \right\}
        |
        Parameters
        ----------
        vector : array-like, shape=[..., d, n, n]
            Vector.
        base_point : array-like, shape=[..., d, n, n]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """

        def _each_is_tangent(vec, bp):
            """Auxiliary function to deal with samples one at a time."""
            for i in range(1, self.d + 1):
                R_i = bp[i - 1]
                Z_i = vec[i - 1]
                cst_1 = gs.all(
                    gs.isclose(
                        Matrices.mul(R_i, Z_i) + Matrices.mul(Z_i, R_i), Z_i, atol=atol
                    )
                )
                cst_2 = gs.all(gs.isclose(Z_i, Matrices.transpose(Z_i), atol=atol))
                cst_3 = gs.all(
                    gs.isclose(
                        Matrices.mul(R_i, Z_i) + Matrices.mul(Z_i, R_i),
                        Matrices.transpose(Z_i),
                        atol=atol,
                    )
                )
                cst_4 = gs.isclose(gs.trace(Z_i), 0.0, atol=atol)
                is_tangent = gs.all([cst_1, cst_2, cst_3, cst_4])
                if not is_tangent:
                    return is_tangent

                for j in range(1, i):
                    R_j = bp[j - 1]
                    Z_j = vec[j - 1]
                    is_tangent = gs.all(
                        gs.isclose(
                            Matrices.mul(Z_i, R_j) + Matrices.mul(R_i, Z_j),
                            gs.zeros((self.n, self.n)),
                            atol=atol,
                        )
                    )
                    if not is_tangent:
                        return is_tangent
            return is_tangent

        if isinstance(base_point, list) or base_point.ndim > 3:
            return gs.stack(
                [_each_is_tangent(vec, bp) for (vec, bp) in zip(vector, base_point)]
            )

        return _each_is_tangent(vector, base_point)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., d, n, n]
            Vector.
        base_point : array-like, shape=[..., d, n, n]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """

        def _each_to_tangent(vec, bp):
            """Auxiliary function to deal with samples one at a time."""
            sym = Matrices.to_symmetric(vec)
            proj = Matrices.mul(bp, Matrices.transpose(bp))
            return Matrices.mul(sym, proj) \
                   + Matrices.mul(proj, sym) \
                   - Matrices.mul(proj, gs.sum(Matrices.mul(sym, proj), axis=0)) \
                   - Matrices.mul(gs.sum(Matrices.mul(proj, sym), axis=0), proj)  #
            # caution: maybe expand_dims
            # Matrices.bracket(sym, proj)+ Matrices.bracket(proj, gs.sum(
            # Matrices.mul(sym, proj), axis=0))

        if isinstance(base_point, list) or base_point.ndim > 3:
            return gs.stack(
                [_each_to_tangent(vec, bp) for (vec, bp) in zip(vector, base_point)]
            )

        return _each_to_tangent(vector, base_point)

    def random_uniform(self, n_samples=1):
        r"""Sample random points from a uniform distribution.

        Drawing from **[Chikuse03]**, **Theorem 1.5.5**, :math: `n_samples \times n
        \times n` scalars are sampled from a standard normal distribution and reshaped
        to :math: `n \times n` matrices. The Polar decomposition of those matrices
        gives unitary matrices that follow a uniform distribution on :math: `V_{n, n}`.
        The unitary matrices columns are then cut in blocks of shapes indicated by
        the increment of the flag manifold index. The uniform columns, that span
        uniformly distributed nested subspaces, are finally transformed into their
        associated projection matrices.

        The proof still needs to be written somewhere (maybe in an incoming paper).

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
        """
        points = gs.random.normal(size=(n_samples, self.n, self.n))
        u = gs.array([polar(point)[0] for point in points])
        projector = []
        for i in range(self.d):
            v_i = u[:, :, self.extended_index[i]: self.extended_index[i + 1]]
            projector.append(Matrices.mul(v_i, Matrices.transpose(v_i)))
        projector = gs.transpose(gs.array(projector), axes=(1, 0, 2, 3))
        return projector[0] if n_samples == 1 else projector

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample random points from a uniform distribution.

        Drawing from **[Chikuse03]**, **Theorem 1.5.5**, :math: `n_samples \times n
        \times n` scalars are sampled from a standard normal distribution and reshaped
        to :math: `n \times n` matrices. The Polar decomposition of those matrices
        gives unitary matrices that follow a uniform distribution on :math: `V_{n, n}`.
        The unitary matrices columns are then cut in blocks of shapes indicated by
        the increment of the flag manifold index. The uniform columns, that span
        uniformly distributed nested subspaces, are finally transformed into their
        associated projection matrices.

        The proof still needs to be written somewhere (maybe in an incoming paper).

        Parameters
        ----------
        n_samples : int
            The number of points to sample
            Optional. default: 1.

        Returns
        -------
        projectors : array-like, shape=[..., n, n]
            Points following a uniform distribution.

        """
        return self.random_uniform(n_samples)


if __name__ == "__main__":
    flag = Flag(n=5, index=[1, 3, 4])
    vector = gs.random.rand(10, flag.d, flag.n, flag.n)
    base_point = flag.random_uniform(n_samples=10)
    print(flag.belongs(base_point))
    print(flag.is_tangent(vector, base_point))
    tangent_vec = flag.to_tangent(vector, base_point)
    print(flag.is_tangent(tangent_vec, base_point))
    print(flag.is_tangent(gs.zeros((10, flag.d, flag.n, flag.n)), base_point))
