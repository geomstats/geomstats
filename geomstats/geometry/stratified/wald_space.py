r"""Classes for the Wald Space and elements therein of class Wald and helper classes.

Lead author: Jonas Lueg

References
----------
.. [Garba21] Garba, M. K., T. M. W. Nye, J. Lueg and S. F. Huckemann.
    "Information geometry for phylogenetic trees"
    Journal of Mathematical Biology, 82(3):19, February 2021a.
    https://doi.org/10.1007/s00285-021-01553-x.
.. [Lueg21] Lueg, J., M. K. Garba, T. M. W. Nye, S. F. Huckemann.
    "Wald Space for Phylogenetic Trees."
    Geometric Science of Information, Lecture Notes in Computer Science,
    pages 710–717, Cham, 2021.
    https://doi.org/10.1007/978-3-030-80209-7_76.
"""

import random
from abc import ABC

import geomstats.backend as gs
from geomstats.geometry.hermitian_matrices import powermh
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.stratified.point_set import (
    Point,
    PointBatch,
    PointSet,
    PointSetMetric,
)
from geomstats.geometry.stratified.trees import (
    ForestTopology,
    Split,
    delete_splits,
    generate_splits,
)
from geomstats.geometry.stratified.vectorization import (
    _manipulate_output,
    broadcast_lists,
    vectorize_point,
)
from geomstats.numerics.interpolation import LinearInterpolator1D
from geomstats.numerics.optimization import ScipyMinimize


def _manipulate_input_with_array(arg, name):
    if gs.is_array(arg):
        ndim = 2 if name == "ambient_point" else 1
        if arg.ndim > ndim:
            return arg, False
        return gs.expand_dims(arg, axis=0), True

    if not isinstance(arg, (list, tuple)):
        return [arg], True

    return arg, False


def _manipulate_output_wald(out, to_list):
    return _manipulate_output(out, to_list, manipulate_output_iterable=WaldBatch)


def make_splits(n_labels):
    """Generate all possible splits of a collection.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    Returns
    -------
    split_iterator : Iterator[Split]
    """
    if n_labels <= 1:
        raise ValueError("`n_labels` must be greater than 1.")
    if n_labels == 2:
        yield Split(part1=[0], part2=[1])
    else:
        for split in make_splits(n_labels=n_labels - 1):
            yield Split(part1=split.part1, part2=split.part2.union((n_labels - 1,)))
            yield Split(part1=split.part1.union((n_labels - 1,)), part2=split.part2)
        yield Split(part1=list(range(n_labels - 1)), part2=[n_labels - 1])


def make_topologies(n_labels):
    """Generate all possible sets of compatible splits of a collection.

    This only works well for `len(n_labels) < 8`.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    Returns
    -------
    topology_iterator : Iterator[ForestTopology]
    """
    if n_labels <= 1:
        raise ValueError("The collection must have 2 elements or more.")
    if n_labels in (2, 3):
        yield ForestTopology(
            partition=(tuple(range(n_labels)),),
            split_sets=(list(make_splits(n_labels)),),
        )
    else:
        pendant_split = Split(part1=[n_labels - 1], part2=list(range(n_labels - 1)))
        for st in make_topologies(n_labels - 1):
            for s in st.split_sets[0]:
                new_split_set = [pendant_split]
                a, b = set(s.part1), set(s.part2)
                for t in st.split_sets[0]:
                    _, d = set(t.part1), set(t.part2)
                    if t != s:
                        if a.issubset(d) or b.issubset(d):
                            new_split_set.append(
                                Split(
                                    part1=t.part1, part2=t.part2.union((n_labels - 1,))
                                )
                            )
                        else:
                            new_split_set.append(
                                Split(
                                    part1=t.part2, part2=t.part1.union((n_labels - 1,))
                                )
                            )
                    else:
                        new_split_set.append(
                            Split(part1=s.part1, part2=s.part2.union((n_labels - 1,)))
                        )
                        new_split_set.append(
                            Split(part1=s.part2, part2=s.part1.union((n_labels - 1,)))
                        )
                yield ForestTopology(
                    partition=(tuple(range(n_labels)),),
                    split_sets=(new_split_set,),
                )


def _generate_partition(n_labels, p_new):
    r"""Generate a random partition of :math:`\{0,\dots,n-1\}`.

    This algorithm works as follows: Start with a single set containing zero,
    then successively add the labels from 1 to n-1 to the partition in the
    following manner: for each label u, with probability `probability`, add the
    label u to a random existing set of the partition, else add a new singleton
    set {u} to the partition (i.e. with probability 1 - `probability`).

    Parameters
    ----------
    p_new : float
        A float between 0 and 1, the probability that no new component is added,
        and 1 - probability that a new component is added.

    Returns
    -------
    partition : list[list[int]]
        A partition of the set :math:`\{0,\dots,n-1\}` into non-empty sets.
    """
    _partition = [[0]]
    for u in range(1, n_labels):
        if gs.random.rand(1) < p_new:
            index = random.randint(0, len(_partition) - 1)
            _partition[index].append(u)
        else:
            _partition.append([u])
    return _partition


def generate_random_wald(n_labels, p_keep, p_new, btol=1e-8, check=True):
    """Generate a random instance of class ``Wald``.

    Parameters
    ----------
    n_labels : int
        The number of labels the wald is generated with respect to.
    p_keep : float
        The probability will be inserted into the generation of a partition as
        well as for the generation of a split set for the topology of the wald.
    p_new : float
        A float between 0 and 1, the probability that no new component is added,
        and probability of 1 - p_new_ that a new component is added.
    btol: float
        Tolerance for the boundary of the coordinates in each grove. Defaults to
        1e-08.
    check : bool
        If True, checks if splits still separate all labels. In this case, the split
        will not be deleted. If False, any split can be randomly deleted.

    Returns
    -------
    random_wald : Wald
        The randomly generated wald.
    """
    partition = _generate_partition(n_labels=n_labels, p_new=p_new)
    split_sets = [generate_splits(labels=_part) for _part in partition]

    split_sets = [
        delete_splits(splits=splits, labels=part, p_keep=p_keep, check=check)
        for part, splits in zip(partition, split_sets)
    ]

    top = ForestTopology(partition=partition, split_sets=split_sets)
    x = gs.random.uniform(size=(top.n_splits,), low=0, high=1)
    x = gs.minimum(gs.maximum(btol, x), 1 - btol)
    return Wald(topology=top, weights=x)


class Wald(Point):
    r"""A class for wälder, that are phylogenetic forests, elements of the Wald Space.

    A wald is essentially a phylogenetic forest with weights between zero and one on
    the edges. The forest structure is stored as a ``ForestTopology`` and the edge
    weights are an array of length that is equal to the total number of splits in the
    structure. These elements are the points in Wald space and other phylogenetic forest
    spaces, like BHV space, although the partition is just the whole set of labels in
    this case.

    Parameters
    ----------
    topology : ForestTopology
        The structure of the forest.
    weights : array-like, shape=[n_splits]
        The edge weights, array of floats between 0 and 1, with m entries, where m is
        the total number of splits/edges in the structure ``top``.
    corr : array-like, shape=[n_labels, n_labels]
        Correlation matrix of the topology with edge weights.
    """

    def __init__(self, topology, weights):
        super().__init__()
        self.topology = topology
        self.weights = weights

        self.corr = self.topology.corr(weights)

    def __repr__(self):
        """Return the string representation of the wald.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_wald : str
            Return the string representation of the wald.
        """
        return repr((self.topology, tuple(self.weights)))

    def __str__(self):
        """Return the fancy printable string representation of the wald.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_wald : str
            Return the fancy readable string representation of the wald.
        """
        return f"({str(self.topology)};{str(self.weights)})"

    def _equal_single(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Wald
            Point to compare against point.
        atol : float

        Returns
        -------
        is_equal : bool
        """
        if self.topology != point.topology:
            return False

        return gs.all(gs.abs(self.weights - point.weights) < atol)

    @vectorize_point((1, "point"))
    def equal(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Wald or WaldBatch
            Point to compare against point.
        atol : float

        Returns
        -------
        is_equal : array-like, shape=[...]
        """
        return gs.array([self._equal_single(point_, atol) for point_ in point])


class WaldBatch(PointBatch):
    """Wald batch."""

    @property
    def topology(self):
        """Forest topology.

        Returns
        -------
        topology : list[ForestTopology]
        """
        return [point.topology for point in self]

    @property
    def weights(self):
        """Edge weights.

        Returns
        -------
        weights : array-like, shape=[n_points, n_splits]
        """
        return gs.stack([point.weights for point in self])

    @property
    def corr(self):
        """Correlation matrix of the topology with edge weights.

        Returns
        -------
        corr : array-like, shape=[n_points, n_nodes, n_nodes]
        """
        return gs.stack([point.corr for point in self])


class WaldSpace(PointSet):
    r"""Class for the Wald space, a metric space for phylogenetic forests.

    A topological space. Points in Wald space are instances of the class :class:`Wald`:
    phylogenetic forests with edge weights between 0 and 1.
    In particular, Wald space is a stratified space, each stratum is called grove.
    The highest dimensional groves correspond to fully resolved or binary trees.
    The topology is obtained from embedding wälder into the ambient space of strictly
    positive :math:`n\times n` symmetric matrices, implemented in the
    class :class:`spd.SPDMatrices`.

    Parameters
    ----------
    n_labels : int
        Integer determining the number of labels in the forests, and thus the shape of
        the correlation matrices: n_labels x n_labels.

    Attributes
    ----------
    ambient_space : Manifold
        The ambient space, the positive definite n_labels x n_labels matrices that the
        WaldSpace is embedded into.
    """

    def __init__(self, n_labels, ambient_space=None, equip=True):
        super().__init__(equip)
        self.n_labels = n_labels

        if ambient_space is None:
            ambient_space = SPDMatrices(n=self.n_labels)
        self.ambient_space = ambient_space

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return WaldSpaceMetric

    def _belongs_single(self, point, atol=gs.atol):
        """Check if a point belongs to Tree space.

        Parameters
        ----------
        point : Wald
            The point to be checked.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : bool
            Boolean denoting if point belongs to wald space.
        """
        if not self.ambient_space.belongs(self.lift(point)):
            return False

        if gs.all(point.weights > 0) and gs.all(point.weights < 1):
            return True

        return False

    @vectorize_point((1, "point"))
    def belongs(self, point, atol=gs.atol):
        """Check if a point `wald` belongs to Wald space.

        From FUTURE PUBLICATION we know that the corresponding matrix of a wald is
        strictly positive definite if and only if the labels are separated by at least
        one edge, which is exactly when the wald is an element of the Wald space.

        Parameters
        ----------
        point : Wald or WaldBatch
            The point to be checked.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...]
            Boolean denoting if `point` belongs to Wald space.
        """
        return gs.array([self._belongs_single(point_, atol) for point_ in point])

    def random_point(self, n_samples=1, p_tree=0.9, p_keep=0.9, btol=1e-8):
        """Sample a random point in Wald space.

        Parameters
        ----------
        n_samples : int
            Number of samples. Defaults to 1.
        p_tree : float between 0 and 1
            The probability that the sampled point is a tree, and not a forest. If the
            probability is equal to 1, then the sampled point will be a tree.
            Defaults to 0.9.
        p_keep : float between 0 and 1
            The probability that a sampled edge is kept and not deleted randomly.
            To be precise, it is not exactly the probability, as some edges cannot be
            deleted since the requirement that two labels are separated by a split might
            be violated otherwise.
            Defaults to 0.9
        btol: float
            Tolerance for the boundary of the coordinates in each grove. Defaults to
            1e-08.

        Returns
        -------
        samples : Wald or WaldBatch
            Points sampled in Wald space.
        """
        p_new = p_tree ** (1 / (self.n_labels - 1))
        forests = [
            generate_random_wald(self.n_labels, p_keep, p_new, btol, check=True)
            for _ in range(n_samples)
        ]

        if n_samples == 1:
            return forests[0]

        return WaldBatch(forests)

    def random_grove_point(self, topology, n_samples=1):
        """Sample a random point in a given grove of wald spcae.

        Parameters
        ----------
        topology : ForestTopology
        n_samples : int
            Number of samples. Defaults to 1.

        Returns
        -------
        samples : Wald or WaldBatch
            Points sampled in Wald space.
        """
        n_splits = topology.n_splits
        weights = gs.random.uniform(size=(n_samples, n_splits))

        forests = [Wald(topology, weights_) for weights_ in weights]

        if n_samples == 1:
            return forests[0]

        return WaldBatch(forests)

    def lift(self, point):
        """Lift a point to the ambient space.

        Parameters
        ----------
        point : Wald or WaldBatch
            The point to be lifted.

        Returns
        -------
        lifted_point : array-like, shape=[..., n_labels, n_labels]
            Point in the ambient space.
        """
        return point.corr


class WaldSpaceMetric(PointSetMetric):
    """Wald space metric.

    Parameters
    ----------
    space : WaldSpace
        Set to equip with metric.
    projection_solver : ProjectionSolver
        Numerical solver to solve projection problem.
    geodesic_solver : GeodesicSolver
        Numerical solver to solve geodesic problem.
    """

    def __init__(self, space, projection_solver=None, geodesic_solver=None):
        super().__init__(space)

        if projection_solver is None:
            projection_solver = LocalProjectionSolver(space)

        if geodesic_solver is None:
            geodesic_solver = SuccessiveProjectionGeodesicSolver(space)

        self.projection_solver = projection_solver
        self.geodesic_solver = geodesic_solver

    def dist(self, point_a, point_b):
        """Distance between two points in the WaldSpace.

        Parameters
        ----------
        point_a: Wald or WaldBatch
            Point in the WaldSpace.
        point_b: Wald or WaldBatch
            Point in the WaldSpace.

        Returns
        -------
        distance : array-like, shape=[...]
            Distance.
        """
        discrete_path = self.discrete_geodesic(point_a, point_b)
        is_list = not isinstance(discrete_path, WaldBatch)
        if not is_list:
            discrete_path = [discrete_path]

        dists = [
            gs.sum(
                self._space.ambient_space.metric.dist(
                    discrete_path_.corr[1:], discrete_path_.corr[:-1]
                ),
                axis=-1,
            )
            for discrete_path_ in discrete_path
        ]

        if is_list:
            return gs.stack(dists)

        return dists[0]

    def geodesic(self, initial_point, end_point):
        """Compute the geodesic in the WaldSpace.

        Parameters
        ----------
        initial_point: Wald or WaldBatch
            Point in the WaldSpace.
        end_point: Wald or WaldBatch
            Point in the WaldSpace.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve.
        """
        return self.geodesic_solver.geodesic(initial_point, end_point)

    def discrete_geodesic(self, initial_point, end_point):
        """Compute a discrete geodesic in the WaldSpace.

        Parameters
        ----------
        initial_point: Wald or WaldBatch
            Point in the WaldSpace.
        end_point: Wald or WaldBatch
            Point in the WaldSpace.

        Returns
        -------
        geod_points : WaldBatch or list[WaldBatch]
            Time parameterized geodesic curve.
        """
        return self.geodesic_solver.discrete_geodesic(initial_point, end_point)

    def projection(self, ambient_point, topology, initial_weights=None):
        """Projects a point into Wald space.

        Parameters
        ----------
        ambient_point : array-like, shape=[..., n_nodes, n_nodes]
            Ambient point to project.
        topology : ForestTopology or list[ForestTopology]
            Stratum topology.
        initial_weights : array-like, shape=[..., n_splits]
            Initial guess for weights.
        """
        return self.projection_solver.projection(
            ambient_point, topology, initial_weights=initial_weights
        )


def _squared_dist_and_grad_affine(space, topology, ambient_point):
    """Squared distance and gradient wrt weights.

    See section 5.1 of [Garba2021]_.

    Parameters
    ----------
    space : WaldSpace
    topology : ForestTopology
    ambient_point : array-like, shape=[n_nodes, n_nodes]
        Point wrt measure distance.

    Returns
    -------
    value_and_grad : callable
        A callable that takes weights and outputs value and grad.
    """
    sqrt_ambient_point, inv_sqrt_ambient_point = powermh(
        ambient_point, [1.0 / 2, -1.0 / 2]
    )

    def _value_and_grad(weights):
        corr = topology.corr(weights)
        inv_corr = gs.linalg.inv(corr)
        grad = topology.corr_gradient(weights)

        target = space.ambient_space.metric.squared_dist(corr, ambient_point)

        target_grad = 2 * gs.trace(
            Matrices.mul(
                gs.linalg.logm(
                    Matrices.mul(inv_sqrt_ambient_point, corr, inv_sqrt_ambient_point)
                ),
                sqrt_ambient_point,
                inv_corr,
                grad,
                inv_sqrt_ambient_point,
            )
        )

        return target, target_grad

    return _value_and_grad


def _squared_dist_and_grad_euclidean(space, topology, ambient_point):
    """Squared distance and gradient wrt weights.

    Parameters
    ----------
    space : WaldSpace
    topology : ForestTopology
    ambient_point : array-like, shape=[n_nodes, n_nodes]
        Point wrt measure distance.

    Returns
    -------
    value_and_grad : callable
        A callable that takes weights and outputs value and grad.
    """

    def _value_and_grad(weights):
        corr = topology.corr(weights)
        grad = topology.corr_gradient(weights)

        target = space.ambient_space.metric.squared_dist(corr, ambient_point)
        target_grad = 2 * gs.sum((corr - ambient_point) * grad, axis=(-2, -1))
        return target, target_grad

    return _value_and_grad


def _squared_dist_and_grad_autodiff(space, topology, ambient_point):
    """Squared distance and gradient wrt weights.

    Parameters
    ----------
    space : WaldSpace
    topology : ForestTopology
    ambient_point : array-like, shape=[n_nodes, n_nodes]
        Point wrt measure distance.

    Returns
    -------
    value_and_grad : callable
        A callable that takes weights and outputs value and grad.
    """

    def _value(weights):
        corr = topology.corr(weights)
        return space.ambient_space.metric.squared_dist(corr, ambient_point)

    return gs.autodiff.value_and_grad(_value)


_AMBIENT_METRIC_TO_SQUARED_DIST_GRAD = {
    "SPDAffineMetric": _squared_dist_and_grad_affine,
    "SPDEuclideanMetric": _squared_dist_and_grad_euclidean,
}


class LocalProjectionSolver:
    """Local projection solver."""

    def __init__(self, space, btol=1e-10):
        self._space = space
        self.btol = btol
        self.optimizer = ScipyMinimize(
            method="L-BFGS-B",
            tol=gs.atol,
        )

    def _get_bounds(self, n_splits):
        return [(self.btol, 1 - self.btol)] * n_splits

    def _projection_single(self, ambient_point, topology, initial_weights=None):
        """Project ambient point into wald space.

        Parameters
        ----------
        ambient_point : array-like, shape=[n_nodes, n_nodes]
            Ambient point to project.
        topology : ForestTopology
            Stratum topology.
        initial_weights : array-like, shape=[n_splits]
            Initial guess for weights.
        """
        if len(topology.partition) == topology.n_labels:
            return Wald(topology=topology, weights=gs.ones(self.n_labels))

        value_and_grad = _AMBIENT_METRIC_TO_SQUARED_DIST_GRAD.get(
            self._space.ambient_space.metric.__class__.__name__,
            _squared_dist_and_grad_autodiff,
        )(
            space=self._space,
            ambient_point=ambient_point,
            topology=topology,
        )

        n_splits = topology.n_splits

        initial_weights = (
            gs.ones(n_splits) * 0.5 if initial_weights is None else initial_weights
        )

        self.optimizer.bounds = self._get_bounds(n_splits)
        res = self.optimizer.minimize(value_and_grad, initial_weights, fun_jac=True)

        if res.status != 0:
            raise ValueError("Projection failed!")

        weights = gs.array(
            [
                _x if self.btol < _x < 1 - self.btol else 0 if _x <= self.btol else 1
                for _x in res.x
            ]
        )

        return Wald(topology=topology, weights=weights)

    @vectorize_point(
        (1, "ambient_point"),
        (2, "topology"),
        (3, "initial_weights"),
        manipulate_input=_manipulate_input_with_array,
        manipulate_output=_manipulate_output_wald,
    )
    def projection(self, ambient_point, topology, initial_weights=None):
        """Project ambient point into wald space.

        Parameters
        ----------
        ambient_point : array-like, shape=[..., n_nodes, n_nodes]
            Ambient point to project.
        topology : ForestTopology or list[ForestTopology]
            Stratum topology.
        initial_weights : array-like, shape=[..., n_splits]
            Initial guess for weights.
        """
        ambient_point, topology, initial_weights = broadcast_lists(
            ambient_point, topology, initial_weights
        )
        return [
            self._projection_single(ambient_point_, topology_, initial_weights_)
            for ambient_point_, topology_, initial_weights_ in zip(
                ambient_point, topology, initial_weights
            )
        ]


class BasicWaldGeodesicSolver(ABC):
    """Abstract class for wald geodesic solver.

    Parameters
    ----------
    space : WaldSpace
    """

    def __init__(self, space):
        self._space = space

    @vectorize_point(
        (1, "initial_point"),
        (2, "end_point"),
        manipulate_output=lambda out, to_list: _manipulate_output(
            out, to_list, manipulate_output_iterable=lambda x: x
        ),
    )
    def discrete_geodesic(self, initial_point, end_point):
        """Compute a discrete geodesic in the WaldSpace.

        Parameters
        ----------
        initial_point: Wald or WaldBatch
            Point in the WaldSpace.
        initial_point: Wald or WaldBatch
            Point in the WaldSpace.

        Returns
        -------
        geod_points : WaldBatch or list[WaldBatch]
            Time parameterized geodesic curve.
        """
        initial_point, end_point = broadcast_lists(initial_point, end_point)
        return [
            self._discrete_geodesic_single(
                initial_point_,
                end_point_,
            )
            for initial_point_, end_point_ in zip(initial_point, end_point)
        ]

    def geodesic(self, initial_point, end_point):
        """Compute the geodesic in the WaldSpace.

        Parameters
        ----------
        initial_point: Wald or WaldBatch
            Point in the WaldSpace.
        initial_point: Wald or WaldBatch
            Point in the WaldSpace.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve.
        """

        def _vec(t, fncs):
            if len(fncs) == 1:
                return fncs[0](t)

            return [fnc(t) for fnc in fncs]

        discrete_path = self.discrete_geodesic(initial_point, end_point)

        if isinstance(discrete_path, WaldBatch):
            return DiscreteWaldPath(self._space, discrete_path)

        return lambda t: _vec(
            t,
            fncs=[
                DiscreteWaldPath(self._space, discrete_path_)
                for discrete_path_ in discrete_path
            ],
        )


class NaiveProjectionGeodesicSolver(BasicWaldGeodesicSolver):
    """Naive geodesic projection solver.

    Implementation of algorithm 1 from [Lueg21]_.
    """

    def __init__(self, space, n_grid=10):
        super().__init__(space)
        self.n_grid = n_grid

    def _discrete_geodesic_single(self, initial_point, end_point):
        """Compute a discrete geodesic in the WaldSpace.

        Parameters
        ----------
        initial_point: Wald
            Point in the WaldSpace.
        end_point: Wald
            Point in the WaldSpace.

        Returns
        -------
        geod_points : WaldBatch
            Time parameterized geodesic curve.
        """
        if initial_point.topology != end_point.topology:
            raise ValueError("Can only handle points in the same grove.")

        topology = initial_point.topology

        initial_corr = self._space.lift(initial_point)
        end_corr = self._space.lift(end_point)

        ambient_geod_func = self._space.ambient_space.metric.geodesic(
            initial_corr, end_point=end_corr
        )

        time = gs.linspace(0, 1, self.n_grid)[1:-1]
        mid_ambient_point = ambient_geod_func(time)

        mid_points = self._space.metric.projection(mid_ambient_point, topology)

        return WaldBatch([initial_point] + mid_points + [end_point])


class SuccessiveProjectionGeodesicSolver(BasicWaldGeodesicSolver):
    """Successive projection geodesic projection solver.

    Implementation of algorithm 2 from [Lueg21]_.
    """

    def __init__(self, space, n_grid=10):
        super().__init__(space)
        self.n_grid = n_grid

    def _discrete_geodesic_single(self, initial_point, end_point):
        """Compute a discrete geodesic in the WaldSpace.

        Parameters
        ----------
        initial_point: Wald
            Point in the WaldSpace.
        end_point: Wald
            Point in the WaldSpace.

        Returns
        -------
        geod_points : WaldBatch
            Time parameterized geodesic curve.
        """
        if initial_point.topology != end_point.topology:
            raise ValueError("Can only handle points in the same grove.")

        topology = initial_point.topology

        left_points = [initial_point]
        right_points = [end_point]
        for i in range(2, self.n_grid // 2 + 1):
            time = 1 / (self.n_grid - i + 1)

            left_corr = self._space.lift(left_points[-1])
            right_corr = self._space.lift(right_points[-1])

            ambient_geod_func = self._space.ambient_space.metric.geodesic(
                left_corr, right_corr
            )

            left_points.append(
                self._space.metric.projection(
                    ambient_geod_func(time)[0],
                    topology,
                    initial_weights=left_points[-1].weights,
                )
            )
            right_points.append(
                self._space.metric.projection(
                    ambient_geod_func(1 - time)[0],
                    topology,
                    initial_weights=right_points[-1].weights,
                )
            )

        right_points.reverse()

        if gs.mod(self.n_grid, 2) == 0:
            return WaldBatch(left_points + right_points)

        left_corr = self._space.lift(left_points[-1])
        right_corr = self._space.lift(right_points[0])

        ambient_geod_func = self._space.ambient_space.metric.geodesic(
            left_corr, end_point=right_corr
        )

        mid_point = self._space.metric.projection(ambient_geod_func(0.5), topology)
        return WaldBatch(left_points + mid_point + right_points)


class DiscreteWaldPath:
    """A uniformly-sampled discrete path.

    Parameters
    ----------
    space : WaldSpace
    path : WaldBatch
        Wald collection with common topology.
    interpolator : Interpolator1D
    """

    def __init__(self, space, path, interpolator=None, **interpolator_kwargs):
        # NB: assumes common topology
        self._topology = path[0].topology

        times = self._get_times(space, path)

        if interpolator is None:
            interpolator = LinearInterpolator1D(
                times, path.weights, point_ndim=1, **interpolator_kwargs
            )
        self.interpolator = interpolator

    def _get_times(self, space, path):
        """Compute times for linear interpolation."""
        dists = space.ambient_space.metric.dist(path[1:].corr, path[:-1].corr)
        cum_dists = gs.cumsum(dists)
        return gs.concatenate([gs.array([0.0]), cum_dists / cum_dists[-1]])

    def __call__(self, t):
        """Interpolate path.

        Parameters
        ----------
        t : array-like, shape=[n_time]
            Interpolation time.

        Returns
        -------
        path : WaldBatch
        """
        weights = self.interpolator(t)
        return WaldBatch([Wald(self._topology, weights_) for weights_ in weights])
