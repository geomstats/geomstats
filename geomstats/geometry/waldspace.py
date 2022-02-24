r"""Wald space.

The metric space :math:`(\mathcal{W}, d_{\mathcal{W}})` obtained from embedding the
phylogenetic forests with :math:`n` labels into the Riemannian manifold of strictly
positive real symmetric :math:`n\times n` matrices with affine-invariant geometry, that
is :class:`spd.SpdMetricAffine`.

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
import math

import geomstats.backend as gs
import geomstats.geometry.spd_matrices as spd
from geomstats.geometry.trees import Split, Structure, Wald


class WaldSpace:
    """Class for the Wald space, a metric space for phylogenetic forests.

    Parameters
    ----------
    n : int
        Integer determining the number of labels in the forests, and thus the shape of
        the correlation matrices: n x n.
    """

    def __init__(self, n):
        self.n = n  # dimension of the wald space
        self.a = spd.SPDMatrices(n=self.n)

    def to_forest(self, point):
        """Take an array [n_samples, n, n] and give a tuple of shape [n_samples]."""

        def _to_forest(_point):
            """Take an array [n, n] and gives back an element of class Wald."""
            _dist = gs.maximum(0, -gs.log(_point))
            _st = compute_structure_from_dist(dist=_dist, btol=10**-10)
            _ells = gs.array(
                [
                    compute_length_of_split_from_dist(sp, dist=_dist)
                    for sp in _st.unravel(_st.split_sets)
                ]
            )
            _x = gs.maximum(0, gs.minimum(1, 1 - gs.exp(-_ells)))
            return Wald(n=self.n, st=_st, x=_x)

        if len(point.shape) == 2:
            return _to_forest(_point=point)
        return tuple([_to_forest(_point=point[i, :, :]) for i in range(point.shape[0])])

    def belongs(self, point, atol=gs.atol):
        """Check if a point `wald` belongs to Wald space.

        Parameters
        ----------
        point : array-like, shape = [n_samples, n, n]
            The point to be checked.
        atol : float
            The tolerance for checking whether a matrix belongs to wald space.

        Returns
        -------
        belongs : bool
            Boolean denoting if `wald` belongs to Wald space.
        """

        def _belongs(p):
            """Take a point (an array) and checks whether it belongs to WaldSpace."""
            if not self.a.belongs(p):
                return False
            for i in range(self.n):
                if p[i, i] != 1:
                    return False
            for i, j, k in it.combinations(range(self.n), 3):
                if p[i, j] < p[i, k] * p[j, k] - atol:
                    return False
            for i, j, k, l in it.combinations(range(self.n), 4):
                if p[i, j] * p[k, l] < min(p[i, k] * p[j, l], p[i, l] * p[j, k]) - atol:
                    return False
            return True

        if len(point.shape) == 2:
            return _belongs(p=point)
        return gs.array([_belongs(w) for w in point])

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
        samples : list, shape=[n_samples, n, n]
            Points sampled in Wald space.
        """

        def _generate_partition(_prob):
            """Generate a random partition of ``range(self.n)``."""
            # start with sampling the connected components.
            _partition = [[0]]
            for u in range(1, self.n):
                # decide whether new component is constructed
                if gs.random.rand(1) < _prob:
                    # add label to random existing component
                    comp_idx = int(gs.random.randint(0, len(_partition), (1,)))
                    _partition[comp_idx].append(u)
                else:
                    # construct new component
                    _partition.append([u])
            return _partition

        def _generate_splits(labels):
            """Generate random maximal set of compatible splits of set ``labels``."""
            # there are no splits of sets consisting of one element
            if len(labels) <= 1:
                return ()
            labels = labels.copy()
            _u = labels.pop(int(gs.random.randint(0, len(labels), (1,))))
            _v = labels.pop(int(gs.random.randint(0, len(labels), (1,))))
            new_labels = [_u, _v]
            # start with the split of the set {_u, _v}
            old_splits = [Split(n=self.n, part1=(_u,), part2=(_v,))]
            # iteratively add new split/edge with leaf _u by docking at random old split
            while labels:
                # random new leaf
                _u = labels.pop(int(gs.random.randint(0, len(labels), (1,))))
                # split representing the pendant edge at the leaf
                new_splits = [Split(n=self.n, part1=(_u,), part2=tuple(new_labels))]
                # random edge that gets divided
                div_ = old_splits.pop(int(gs.random.randint(0, len(old_splits), (1,))))
                # create two parts from old divided edge
                new_splits.append(
                    Split(n=self.n, part1=div_.part1 + (_u,), part2=div_.part2)
                )
                new_splits.append(
                    Split(n=self.n, part1=div_.part1, part2=div_.part2 + (_u,))
                )
                # add the new label _u to every other old split correctly
                for sp in old_splits:
                    new_splits.append(
                        Split(
                            n=self.n,
                            part1=sp.point_away_split(div_),
                            part2=sp.point_to_split(div_) + (_u,),
                        )
                    )
                new_labels.append(_u)
                old_splits = new_splits
            return old_splits

        def _check_separability(splits, labels):
            """Check for existence of at least one split between each pair of labels."""
            return gs.all(
                [
                    gs.any([sp.separates(_u, _v) for sp in splits])
                    for _u, _v in it.combinations(labels, 2)
                ]
            )

        def _delete_random_edges(splits, labels, probability):
            """Delete random splits, given set of ``splits`` of set of ``labels``."""
            if probability == 1:
                return splits
            # random decide whether to delete splits, starting from the last split
            print(f"Test 1 -- splits are {splits}.")
            for i in reversed(range(len(splits))):
                # in this case, delete split, if allowed
                if gs.random.rand(1) > probability:
                    splits_copy = splits.copy()
                    splits_copy.pop(i)
                    # if allowed (i.e. exists split separating every pair of labels),
                    # delete the split, else not (don't change splits)
                    if _check_separability(splits=splits_copy, labels=labels):
                        splits = splits_copy
            print(f"Test 2 -- splits are {splits}.")
            return splits

        def _generate_wald(_prob):
            """Generate a random wald."""
            # generate random partition
            partition = _generate_partition(_prob=_prob)
            # generate random sets of splits for each component of the partition
            split_sets = [_generate_splits(labels=_part) for _part in partition]
            # delete random splits of those sets of splits
            split_sets = [
                _delete_random_edges(splits=_splits, labels=_part, probability=_prob)
                for _part, _splits in zip(partition, split_sets)
            ]
            # create the structure for the wald, that is partition + sets of splits
            st = Structure(n=self.n, partition=partition, split_sets=split_sets)
            # generate random weights for the edges
            x = gs.random.uniform(size=(len(st.unravel(st.split_sets)),), low=0, high=1)
            # TODO element wise minimum of arrays, minimum needed. gs.minimum not impl.
            x = gs.minimum(gs.maximum(btol, x), 1 - btol)
            # create the wald
            return Wald(n=self.n, st=st, x=x)

        # generate samples:
        prob = prob ** (1 / (self.n - 1))
        if n_samples == 1:
            sample = _generate_wald(_prob=prob)
            print(sample)
            print(sample.x.shape)
            return sample.corr
        return gs.array([_generate_wald(_prob=prob).corr for _ in range(n_samples)])


def equivalence_partition(group, relation):
    """Partition a set of objects into equivalence classes.

    Taken from 'https://stackoverflow.com/questions/38924421/is-there-a-standard-way-to-
    partition-an-interable-into-equivalence-classes-given'.

    Parameters
    ----------
    group : iterable
        Collection of objects to be partitioned.
    relation : callable
        Equivalence relation. i.e. relation(o1,o2) evaluates to True
        if and only if o1 and o2 are equivalent.

    Returns
    -------
    classes : iterable
        A sequence of sets. Each one is an equivalence class.
    partitions : dict
        A dictionary mapping objects to equivalence classes.
    """
    classes = []
    partitions = {}
    for o in group:  # for each object
        # find the class it is in
        found = False
        for c in classes:
            if relation(next(iter(c)), o):  # is it equivalent to this class?
                c.add(o)
                partitions[o] = c
                found = True
                break
        if not found:  # it is in a new class
            classes.append({o})
            partitions[o] = classes[-1]
    classes = tuple(map(tuple, classes))
    partitions = {key: tuple(item) for key, item in partitions.items()}
    return classes, partitions


def compute_structure_from_dist(dist, btol=10**-10):
    """Compute the structure induced by a distance matrix, entries = inf possible."""
    _n = dist.shape[0]
    _partition, _ = equivalence_partition(
        group=range(_n), relation=lambda i, j: dist[i, j] < math.inf
    )
    # for each component, compute the splits in that TREE
    _split_collection = [
        _compute_splits_from_sub_dist(sub_dist=dist, sub_labels=labels, btol=btol)
        for labels in _partition
    ]
    return Structure(n=_n, partition=_partition, split_sets=_split_collection)


def _compute_splits_from_sub_dist(sub_dist, sub_labels, btol=10**-10):
    """Compute the splits of a single tree corresponding to distance matrix.

    The label set might be smaller than the dimension of dist, in which case,
    canonically, the sub-matrix is taken.
    """
    _n = sub_dist.shape[0]
    # test if we have only one label, then we have zero splits. note that sub_dist might
    # be bigger!
    if len(sub_labels) == 1:
        return ()

    # plan is the following:
    # generate all pairs, i.e. sets of the form {i, j}
    # compute all 2-splits, i.e. splits of the form {i, j}|{k, l}
    # compute their length and take only those with positive length (i.e. they are
    # compatible with forest structure)
    # the pair that appears most often in those 2-splits must be a cherry
    # the cherry is reduced to a single leaf with a new label and the process is
    # repeated
    # after that, all new labels are resolved back to their original leaves,
    # giving the tree

    used_keys = set(sub_labels)
    key_dict = {u: {u} for u in used_keys}
    # important: pairs is a list for it to be compatible with sorting statements
    pairs = [{u, v} for u, v in it.product(used_keys, repeat=2) if u < v]
    # for technical reasons: we have new keys, and a maximum of 2*N - 3, and need to
    # compare those splits, thus w.r.t. n
    _m = 2 * _n - 3
    pair_splits = {
        Split(n=_m, part1=p1, part2=p2)
        for p1, p2 in it.product(pairs, repeat=2)
        if not set(p1) & set(p2)
    }
    _ps_lengths = {
        sp: compute_length_of_split_from_dist(split=sp, dist=sub_dist)
        for sp in pair_splits
    }
    pair_splits = {sp for sp in pair_splits if _ps_lengths[sp] > btol}

    # start with the pendant edges, although not all of them need to exist!!!
    splits = {Split(n=_m, part1=(u,), part2=tuple(used_keys - {u})) for u in used_keys}
    _s_lengths = {
        sp: compute_length_of_split_from_dist(split=sp, dist=sub_dist) for sp in splits
    }
    splits = {sp for sp in splits if _s_lengths[sp] > btol}

    while pair_splits:
        # compute how often a pair appears, take the most frequent one as the next
        # cherry to reduce
        frequencies = [
            len([sp for sp in pair_splits if sp.contains(pair)]) for pair in pairs
        ]
        # most frequent pair is next cherry (set of two used keys)
        argsort_fr = sorted(range(len(frequencies)), key=frequencies.__getitem__)
        next_cherry = set(pairs[argsort_fr[-1]])
        # the cherry determines a unique split, it's saved in its unresolved state
        # (identify with new key)
        splits = splits.add(
            Split(n=_m, part1=tuple(next_cherry), part2=tuple(used_keys - next_cherry))
        )
        new_key = max(key_dict.keys()) + 1
        # update the pair_splits. some pairs are vanishing (namely those that have the
        # next_cherry as one part).
        pair_splits = {
            Split(
                n=_m,
                part1=tuple([new_key if u in next_cherry else u for u in sp.part1]),
                part2=tuple([new_key if u in next_cherry else u for u in sp.part2]),
            )
            for sp in pair_splits
            if not sp.contains(next_cherry)
        }
        # store the new key that now encodes the two labels contained in the cherry
        key_dict[new_key] = set.union(*[key_dict[k] for k in next_cherry])
        # update the label set with the new key and removing the labels contained in
        # the cherry
        used_keys = used_keys - next_cherry | {new_key}
        # update the pairs that are contained in the set of pair_splits
        pairs = [
            [new_key if u in next_cherry else u for u in p]
            for p in pairs
            if next_cherry != set(p)
        ]
        # eliminate doubles (frozenset is needed since a set must contain hashable
        # types only).
        pairs = list(map(list, set(map(frozenset, pairs))))

    # finally, a star tree is what's left, we add those splits to the tree_splits as
    # well
    # they might very well be already contained in tree_splits
    splits = splits | {
        Split(n=_m, part1=used_keys - {u}, part2=(u,)) for u in used_keys
    }

    # now we can convert the splits consisting of unresolved keys back into splits
    # containing only original labels
    # and again eliminate doubles: that's why we use the set.
    splits = {
        Split(
            n=_n,
            part1=tuple(set.union(*[key_dict[u] for u in sp.part2])),
            part2=tuple(set.union(*[key_dict[u] for u in sp.part1])),
        )
        for sp in splits
    }
    _s_lengths = {
        sp: compute_length_of_split_from_dist(split=sp, dist=sub_dist) for sp in splits
    }
    splits = sorted([sp for sp in splits if _s_lengths[sp] > btol])
    return splits


def compute_length_of_split_from_dist(split: Split, dist):
    """Return the supposed length of a split according to the dist matrix dist."""
    if not split:  # if one part is empty, define this as minus infinity.
        return -math.inf

    pairs1 = list(
        it.combinations(split.part1, 2)
        if len(split.part1) > 1
        else [(split.part1[0],) * 2]
    )
    pairs2 = list(
        it.combinations(split.part2, 2)
        if len(split.part2) > 1
        else [(split.part2[0],) * 2]
    )
    # take care of infinite cases. some splits might have infinite length
    # (although we generally want to avoid that)
    return 0.5 * gs.amin(
        [
            [
                math.inf
                if math.isinf(dist[p1[0], p2[0]]) or math.isinf(dist[p1[1], p2[1]])
                else dist[p1[0], p2[0]] + dist[p1[1], p2[1]] - dist[p1] - dist[p2]
                for p1 in pairs1
            ]
            for p2 in pairs2
        ]
    )


def neighbours(st: Structure, sp: Split):
    """Return structures at the boundary of ``st`` with split ``sp``.

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
    set_ab = set(sp.part1)
    rest_dict = {s: set(s.point_away_split(sp)) for s in st.split_sets[0] if s != sp}
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
