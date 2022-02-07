r""" Wald space.

A metric space :math:`(\mathcal{W}\times, d_{\mathcal{W}})` for phylogenetic forests
obtained from embedding into the strictly positive definite matrices.

Lead author: Jonas Lueg

Points in Wald space are instances of the class :class:`Wald`: phylogenetic forests with
edge weights between 0 and 1.

In particular, Wald space is a stratified space, each stratum is called grove.
The highest dimensional groves correspond to fully resolved or binary trees.

The geometry on the ambient space :class:`Spd` of strictly positive definite matrices is
the so-called affine-invariant geometry, implemented in :class:`spd.SPDMetricAffine`.

References
----------
[Garba21]_  Garba, M. K., T. M. W. Nye, J. Lueg and S. F. Huckemann.
            "Information geometry for phylogenetic trees"
            Journal of Mathematical Biology, 82(3):19, February 2021a.
            https://doi.org/10.1007/s00285-021-01553-x.
[Lueg21]_   Lueg, J., M. K. Garba, T. M. W. Nye, S. F. Huckemann.
            "Wald Space for Phylogenetic Trees."
            Geometric Science of Information, Lecture Notes in Computer Science,
            pages 710–717, Cham, 2021.
            https://doi.org/10.1007/978-3-030-80209-7_76.
"""

import itertools as it
import numpy as np
import scipy.optimize

import geomstats.backend as gs
import geomstats.geometry.spd_matrices as spd
from geomstats.geometry.spd_matrices import SPDMatrices as Spd

from geomstats.geometry.trees import Split, Structure, Wald

from geomstats.geometry.matrices import Matrices as Mat
from geomstats.geometry.general_linear import GeneralLinear as GenL
from geomstats.geometry.symmetric_matrices import SymmetricMatrices as Sym


class WaldSpace(object):
    """ Class for the Wald space, a metric space for phylogenetic forests.

    Parameters
    ----------
    n : int
        Integer determining the number of labels in the forests, and thus the shape of
        the correlation matrices: n x n.
    """

    def __init__(self, n):
        self.n = n  # dimension of the wald space
        self.a = spd.SPDMetricAffine(n=n, power_affine=1)  # a for ambient space

    def belongs(self, wald):
        """Check if a point `wald` belongs to Wald space.

        Parameters
        ----------
        wald : Wald or list of Wald
            The point to be checked.

        Returns
        -------
        belongs : bool
            Boolean denoting if `wald` belongs to Wald space.
        """
        def _belongs_wald(w):
            is_n = (w.n == self.n)
            is_pd = Mat.is_pd(mat=self.lift(w))
            is_in_01 = gs.all(0 <= w.x) and gs.all(w.x <= 1)
            return is_n and is_pd and is_in_01

        if isinstance(wald, Wald):
            return _belongs_wald(w=wald)
        else:
            return gs.array([_belongs_wald(w=_w) for _w in wald])

    def random_point(self, n_samples=1, btol=1e-08, prob=0.9):
        """Sample a random point in Wald space.

        Parameters
        ----------
        n_samples : int
            Number of samples. Defaults to 1.
        btol: float
            Tolerance for the boundary of the coordinates in each grove. Defaults to
            1e-08.
        prob : float in (0, 1)
            The probability that the sampled point is a tree, and not a forest. If the
            probability is equal to 1, then the sampled point will be a fully resolved
            tree. Defaults to 0.9.
        Returns
        -------
        samples : list, shape=[..., n_samples]
            Points sampled in Wald space.
        """

        def _generate_partition(_prob):
            """ Generates a random partition of ``range(self.n)``. """
            # start with sampling the connected components.
            _partition = [[0]]
            for u in range(1, self.n):
                # decide whether new component is constructed
                if gs.random.rand() < _prob:
                    # add label to random existing component
                    _partition[gs.random.randint(len(_partition))].append(u)
                else:
                    # construct new component
                    _partition.append([u])
            return _partition

        def _generate_splits(labels):
            """ Generates random maximal set of compatible splits of set ``labels``. """
            # there are no splits of sets consisting of one element
            if len(labels) <= 1:
                return tuple()
            labels = labels.copy()
            _u = labels.pop(gs.random.randint(len(labels)))
            _v = labels.pop(gs.random.randint(len(labels)))
            new_labels = [_u, _v]
            # start with the split of the set {_u, _v}
            old_splits = [Split(n=self.n, part1=(_u,), part2=(_v,))]
            # iteratively add new split/edge with leaf _u by docking at random old split
            while labels:
                # random new leaf
                _u = labels.pop(gs.random.randint(len(labels)))
                # split representing the pendant edge at the leaf
                new_splits = [Split(n=self.n, part1=(_u,), part2=tuple(new_labels))]
                # random edge that gets divided
                div_split = old_splits.pop(gs.random.randint(len(old_splits)))
                # create two parts from old divided edge
                new_splits.append(Split(n=self.n, part1=div_split.part1 + (_u,),
                                        part2=div_split.part2))
                new_splits.append(Split(n=self.n, part1=div_split.part1,
                                        part2=div_split.part2 + (_u,)))
                # add the new label _u to every other old split correctly
                for sp in old_splits:
                    _part = sp.point_to_split(other=div_split)
                    new_splits.append(
                        Split(n=self.n, part1=sp.point_away_split(div_split),
                              part2=sp.point_to_split(div_split) + (_u,)))
                new_labels.append(_u)
                old_splits = new_splits
            return old_splits

        def _delete_random_edges(splits, labels, probability):
            """ Given set of ``splits`` of set of ``labels``, delete random splits. """
            if probability == 1:
                return splits
            np.random.shuffle(splits)
            # random decide whether to delete splits, starting from the last split
            for i in reversed(range(len(splits))):
                # in this case, delete split, if allowed
                if gs.random.rand() > probability:
                    splits_copy = splits.copy()
                    splits_copy.pop(i)
                    # if allowed (i.e. exists split separating every pair of labels),
                    # delete the split, else not (don't change splits)
                    if _check_separability(splits=splits_copy, labels=labels):
                        splits = splits_copy
            return splits

        def _check_separability(splits, labels):
            """ Checks if exists at least one split between each pair of labels. """
            return gs.all([gs.any([sp.separates(_u, _v) for sp in splits]) for _u, _v in
                           it.combinations(labels, 2)])

        # generate samples:
        prob = prob ** (1 / (self.n - 1))
        samples = []
        for _ in range(n_samples):
            # generate random partition
            partition = _generate_partition(_prob=prob)
            # generate random sets of splits for each component of the partition
            split_sets = [_generate_splits(labels=_part) for _part in partition]
            # delete random splits of those sets of splits
            split_sets = [
                _delete_random_edges(splits=_splits, labels=_part, probability=prob)
                for _part, _splits in zip(partition, split_sets)]
            # create the structure for the wald, that is partition + sets of splits
            st = Structure(n=self.n, partition=partition, split_sets=split_sets)
            # generate random weights for the edges
            x = gs.random.uniform(size=len(st.unravel(st.split_sets)))
            # TODO element wise minimum of arrays, minimum needed. gs.minimum not impl.
            x = np.minimum(np.maximum(btol, x), 1 - btol)
            # create the wald
            samples.append(Wald(n=self.n, st=st, x=x))

        if n_samples is 1:
            return samples[0]
        else:
            return samples

    @staticmethod
    def lift(point):
        """ Lifts a wald into the strictly positive definite matrices.

        Parameters
        ----------
        point : Wald or array-like, shape=[n, n]
            The point to be lifted.

        Returns
        -------
        lifted_point : array-like, shape=[..., n, n]
            The lifted point that is a strictly positive definite matrix.
        """
        if isinstance(point, Wald):
            return point.corr
        else:
            return gs.array([p.corr if isinstance(p, Wald) else p for p in point])

    def lift_vector(self, vector, point: Wald):
        """ Lifts vector in tangent space of grove to vector in ambient tangent space.

        Parameters
        ----------
        vector : array-like, shape=[..., len(p.x)] or array-like, shape=[..., n, n]
            The vector in the tangent space of the grove (dimension = number of edges).
        point : Wald
            The tangent space is at the point p in Wald space.

        Returns
        -------
        lifted_vector : array-like, shape=[..., n, n]
            The lifted vector in the tangent space of SPD matrices at the lifted point.
        """
        _gradient_x = point.st.chart_gradient(x=point.x)

        def _lift_vector(v):
            """ Lifts a single vector. """
            try:
                assert v.shape == self.lift(point=point).shape
                return v
            except AssertionError:
                # TODO: this can be implemented more efficiently, vector arithmetic
                return gs.array(
                    gs.sum([v[i] * b_i for i, b_i in enumerate(_gradient_x)], axis=0))

        try:
            # if it is a list, then apply to all elements
            return gs.array([_lift_vector(_v) for _v in vector])
        except TypeError:
            return _lift_vector(vector)

    def length(self, path):
        """ Approximates the length of a given discretized path in Wald space.

        Parameters
        ----------
        path : list of Wald
            The path with objects of type Wald.

        Returns
        -------
        length : float
            The length of the path, the sum of successive pairwise distances.
        """
        _length = gs.sum(self.a.dist(self.lift(_p), self.lift(_q))
                         for _p, _q in zip(path[:-1], path[1:]))
        return _length

    def geodesic(self, p: Wald, q: Wald, n_points=20, **proj_args):
        """ Approximates a shortest path between ``p`` and ``q``.

        Essentially an implementation of Algorithm 2 from [Lueg21]_, with minor
        corrections made to the time steps.

        Parameters
        ----------
        p : Wald
            The starting point of the geodesic.
        q : Wald
            The end point of the geodesic.
        n_points : int
            The number of desired points on the geodesic.
        **proj_args
            These parameters will be passed to the ``projection_ambient`` function used
            by the algorithm.

        Returns
        -------
        geodesic : list of Wald
            The approximation of a shortest path between ``p`` and ``q``.

        References
        ----------
        [Lueg21]_   Lueg, J., M. K. Garba, T. M. W. Nye, S. F. Huckemann.
                    "Wald Space for Phylogenetic Trees."
                    Geometric Science of Information, Lecture Notes in Computer Science,
                    pages 710–717, Cham, 2021.
                    https://doi.org/10.1007/978-3-030-80209-7_76.
        """
        n_half = n_points // 2
        g, h = [p], [q]
        for i in range(1, n_half):
            path = self.a.geodesic(initial_point=self.lift(g[i - 1]),
                                   end_point=self.lift(h[i - 1]))
            g.append(
                self.projection_ambient(path(t=1 / (n_points - 2 * i + 1)),
                                        st0=g[i - 1].st, x0=g[i - 1].x, **proj_args))
            h.append(
                self.projection_ambient(path(t=1 - 1 / (n_points - 2 * i + 1)),
                                        st0=h[i - 1].st, x0=h[i - 1].x, **proj_args))

        if n_points % 2 != 0:  # if needed, add a point s.t. we have n_points on path.
            path = self.a.geodesic(initial_point=g[-1], end_point=h[-1])
            g.append(self.projection_ambient(path(t=0.5), st0=g[-1].st, x0=g[-1].x,
                                             **proj_args))
        return g + h[::-1]

    def projection_ambient(self, point, st0: Structure, x0=None, method='global',
                           btol=1e-08, ftol=2.22e-09, gtol=1e-05):
        """ Computes the (locally) orthogonal projection from SPD onto Wald space.

        Parameters
        ----------
        point : array-like, shape=[n, n]
            The strictly positive definite matrix that is projected onto the Wald space.
        st0 : Structure
            The tree topology determining on which grove we start searching from.
        x0 : array-like, shape=[len(st0.unravel(st.split_collection))]
            The weights of the edges of ``st0`` we start searching from.
            Defaults to ``None``.
        method : str
            The projection method to use. Either `local` or `global`. If `local`, then
            we project only onto the grove with structure ``st0``. If `global`, we use
            gradient descent to travel through Wald space. Defaults to `global`.
        btol : float
            The tolerance for the boundary of a grove, coordinates that are closer to
            zero or one than `btol` are considered to be zero or one, respectively.
            Defaults to 1e-08.
        ftol : float
            Minimization algorithm within the grove terminates if the relative function
            value descent < ``ftol``. Defaults to 2.22e-09.
        gtol : float
            Minimization algorithm within the grove terminates if the projected gradient
            < ``gtol``. Defaults to 1e-05.
        """
        if method == 'local':
            return self._proj_method_local(point=point, st0=st0, x0=x0, btol=btol,
                                           ftol=ftol, gtol=gtol)
        elif method == 'global':
            return self._proj_method_global(point=point, st0=st0, x0=x0, btol=btol,
                                            ftol=ftol, gtol=gtol)
        else:
            raise NotImplementedError(f"Projection method '{method}' not implemented.")

    def _proj_method_local(self, point, st0: Structure, x0, btol, ftol, gtol):
        """ Projection of a matrix onto the grove with structure ``st0``.

        See ``self.projection_ambient`` for the documentation of the parameters.
        """
        # the case where the grove has only one point: the isolated forest.
        if len(st0.partition) == st0.n:
            st0 = Structure(n=st0.n, partition=tuple(() for _ in st0.n),
                            split_sets=tuple(() for _ in st0.n))
            return Wald(n=st0.n, st=st0, x=gs.array())
        # the target function that we will minimize.
        target_and_gradient = self._proj_target_gradient(point=point, st=st0)
        # the number of edges/splits in the forest
        n_edges = len(st0.unravel(st0.split_sets))
        # default parameters
        if x0 is None:
            x0 = np.repeat(0.5, n_edges)
        bounds = [(btol, 1 - btol)] * n_edges
        # minimize.
        res = scipy.optimize.minimize(target_and_gradient, x0=x0, jac=True,
                                      method='L-BFGS-B', bounds=bounds, tol=None,
                                      options={'gtol': gtol, 'ftol': ftol})
        # if x is close to boundary, treat as if it was on boundary
        # print(res)
        if res.status != 0:
            raise ValueError("Projection failed!")
        x = [_x if btol < _x < 1 - btol else 0 if _x <= btol else 1 for _x in res.x]
        if np.allclose(np.eye(st0.n), Wald(st=st0, x=x, n=st0.n).corr):
            raise UserWarning(
                "The projection projected onto the isolated forest. Might be an error.")
        return Wald(st=st0, x=x, n=st0.n)

    def _proj_method_global(self, point, st0: Structure, x0, btol, ftol, gtol):
        """ Projection of a matrix onto the Wald space via descent type method.

        See ``self.projection_ambient`` for the documentation of the parameters.
        """
        if len(st0.partition) > 1:
            raise NotImplementedError("The grove to start with should be a tree.")

        # back to the algorithm
        ruled_out_splits = set()

        while True:
            wald = self._proj_method_local(point=point, st0=st0, x0=x0, btol=btol,
                                           ftol=ftol, gtol=gtol)
            if np.any(wald.x == 1):
                raise ValueError("Projection onto forest, this case is not treated.")
            if np.all(wald.x != 0):
                # in this case, we are in the interior of a grove, a minimum.
                return wald
            # rule out all splits that have converged to zero (assume we have tree)
            splits0 = [_split for _i, _split in enumerate(st0.split_sets[0])
                       if wald.x[_i] == 0]
            # we assume that only one edge at a time goes to zero
            # TODO: treat also the case where more than one edge goes to 0
            if len(splits0) > 1:
                raise UserWarning("Projection might be wrong, more than one edge is 0.")
            # these are the edges that were already searched and are ruled out
            ruled_out_splits = ruled_out_splits | set(splits0)
            # new candidates // search directions
            candidates = [_st for _st, _sp in neighbours(st=wald.st, sp=splits0[0])
                          if _sp not in ruled_out_splits]
            if len(candidates) == 0:
                return wald

            st_next = candidates[0]
            # give starting vector the values of minimizer in previous grove except for
            # the new edges, there we move a little into the grove.
            x0 = np.array(
                [wald.x[wald.st.where(_s)] if _s in st0.split_sets[0] else 0.4 for
                 _s in st_next.split_sets[0]])
            st0 = st_next

    def _proj_target_gradient(self, point, st: Structure):
        """ Computes the squared distance from ``p`` to point in grove ``st``. 
        
        The target function is the squared distance from the matrix ``p`` to the lifted
        version of the wald with structure ``st`` and coordinates ``x``, where the
        squared distance is with respect to the ambient space of strictly positive
        definite matrices with affine-invariant geometry.
        Furthermore, the gradient of this target function is returned, i.e. a list of
        partial derivatives of the target function in the coordinates.
        
        Parameters
        ----------
        point : array-like, shape=[n, n]
            A positive definite matrix, a point in ambient space.
        st : Structure
            The tree topology of the grove where we consider points in.

        Returns
        -------
        target : callable
            The target function taking coordinates ``x`` as arguments.
        target_gradient : callable
            The gradient function taking coordinates ``x`` as arguments.
        """
        p_sqrt = Sym.powerm(point, power=1.0 / 2)
        p_inv_sqrt = Sym.powerm(point, power=-1.0 / 2)

        def target_gradient(x):
            _corr = st.chart(x)
            _target = self.a.squared_dist(point_a=_corr, point_b=point)
            dummy = Spd.logm(Mat.mul(p_inv_sqrt, _corr, p_inv_sqrt))
            dummy2 = Mat.mul(p_inv_sqrt, dummy, p_sqrt, GenL.inverse(_corr))
            _target_gradient = np.array([0.5 * Mat.trace_product(dummy2, grad)
                                         for grad in st.chart_gradient(x)])
            return _target, _target_gradient

        return target_gradient


def neighbours(st: Structure, sp: Split):
    """ Returns structures at the boundary of ``st`` with split ``sp``.

    Essentially gives the structures that are obtained when a nearest neighbour
    interchange is performed, that are those two fully resolved tree topologies adjacent
    to the one where the split ``sp``is removed.

    Parameters
    ----------
    st : Structure
        The structure whose neighbours are to be determined.
    sp : Split
        The split that is removed, two other splits can be inserted instead.

    Returns
    -------
    neighbours : list of tuple
        A two-element list where each element is a tuple of the form (``st0``, ``sp0``),
        that is the neighboring structure ``st0`` and the corresponding new split
        ``sp0``.
    """
    # assume that split 'sp' splits into (set_a + set_b) vs. (set_c + set_d)
    set_a, set_b, set_c, set_d = set(), set(), set(), set()
    set_ab, set_cd = set(sp.part1), set(sp.part2)
    rest_dict = {s: set(s.point_away_split(sp)) for s in st.split_sets[0]
                 if s != sp}
    empty_count = 4  # stop if all sets (set_a, ..., set_d) are filled with something.
    for s in sorted(rest_dict, key=lambda k: len(rest_dict[k]), reverse=True):
        rest = rest_dict[s]
        if rest.issubset(set_ab):
            if set_a.issubset(rest):
                set_a = rest
                empty_count -= 1
            elif rest.issubset(set_a):
                continue
            elif set_b.issubset(rest):
                set_b = rest
                empty_count -= 1
            elif rest.issubset(set_b):
                continue
        else:
            if set_c.issubset(rest):
                set_c = rest
                empty_count -= 1
            elif rest.issubset(set_c):
                continue
            elif set_d.issubset(rest):
                set_d = rest
                empty_count -= 1
            elif rest.issubset(set_d):
                continue
        if not empty_count:
            break
    sp1 = Split(n=sp.n, part1=set_a | set_c, part2=set_b | set_d)
    sp2 = Split(n=sp.n, part1=set_a | set_d, part2=set_b | set_c)
    split_collection1 = [[sp1] + [s for s in st.split_sets[0] if s != sp]]
    split_collection2 = [[sp2] + [s for s in st.split_sets[0] if s != sp]]
    st1 = Structure(n=st.n, partition=st.partition, split_sets=split_collection1)
    st2 = Structure(n=st.n, partition=st.partition, split_sets=split_collection2)
    return [(st1, sp1), (st2, sp2)]
