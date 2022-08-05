"""Graph Space.

Lead author: Anna Calissano.
"""

import functools
import itertools
from abc import ABCMeta, abstractmethod

import networkx as nx
import scipy

import geomstats.backend as gs
from geomstats.errors import check_parameter_accepted_values
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stratified.point_set import (
    Point,
    PointSet,
    PointSetMetric,
    _vectorize_point,
)


def _pad_graph_points_with_zeros(points, n_nodes, copy=False):
    r"""Pad graphs point with zeros.

    Graph space is an embedding for adjacency matrices of the same dimension. Smaller
    graphs can be padded adding zero nodes and edges, i.e., block of zero rows and
    columns.

    Parameters
    ----------
    points : GraphPoint
        Graph of the original dimension to be augmented.

    n_nodes : int
        A positive number representing the number of desired nodes.

    Returns
    -------
    points : GraphPoint
        Set of Graphs with the new number of nodes.
    """
    if type(points) is GraphPoint:
        if copy:
            points = GraphPoint(points.adj)

        n = n_nodes - points.n_nodes
        if n > 0:
            points.adj = gs.pad(points.adj, [[0, n], [0, n]])
    else:
        points = [
            _pad_graph_points_with_zeros(point, n_nodes, copy=copy) for point in points
        ]

    return points


def _pad_array_with_zeros(array, n_nodes):
    r"""Pad graphs represented as array with zeros.

    Graph space is an embedding for adjacency matrices of the same dimension. Smaller
    graphs can be padded adding zero nodes and edges, i.e., block of zero rows and
    columns.

    Parameters
    ----------
    array : array-like, shape=[n_obs, n_original_nodes, n_original_nodes]
        Adjacency matrices of the original dimension to be augmented.

    n_nodes : int
        A positive number representing the number of desired nodes.

    Returns
    -------
    array : array-like, shape=[n_obs, n_nodes, n_nodes]
        Set of adjacency matrices with the new nr of nodes.
    """
    if array.shape[-1] < n_nodes:
        n = n_nodes - array.shape[-1]
        paddings = [[0, 0]] + [[0, n]] * 2 if array.ndim > 2 else [[0, n]] * 2
        array = gs.pad(array, paddings)

    return array


def _pad_points_with_zeros(points, n_nodes, copy=True):
    r"""Pad graphs with zeros.

    Graph space is an embedding for adjacency matrices of the same dimension. Smaller
    graphs can be padded adding zero nodes and edges, i.e., block of zero rows and
    columns.

    Parameters
    ----------
    points : array-like, shape=[n_obs, n_original_nodes, n_original_nodes] or GraphPoint
        Adjacency matrices or GraphPoint of the original dimension to be augmented.

    n_nodes : int
        A positive number representing the number of desired nodes.

    Returns
    -------
    array : array-like, shape=[n_obs, n_nodes, n_nodes] or GraphPoint
        Set of adjacency matrices or GraphPoint with the new nr of nodes.
    """
    if type(points) in [list, tuple, GraphPoint]:
        points = _pad_graph_points_with_zeros(points, n_nodes, copy=copy)
    else:
        points = _pad_array_with_zeros(points, n_nodes)

    return points


def _vectorize_graph(*args_positions):
    r"""Vectorize GraphPoint or array into array.

    Turns GraphPoint or array into an array of adjacency matrices. This way all
    the methods only need to work for the set of points case (output shape
    should be returned accordingly to the input though).

    Parameters
    ----------
    points : array-like, shape=[n_obs, n_nodes, n_nodes] or GraphPoint
        Adjacency matrix or GraphPoint.

    Returns
    -------
    array : array-like, shape=[n_obs, n_nodes,  n_nodes]
        Array of vectorized adjacency matrices.
    """

    def _manipulate_input(arg):
        if type(arg) not in [list, tuple, GraphPoint]:
            return arg

        if type(arg) is GraphPoint:
            return arg.adj

        return gs.array([graph.adj for graph in arg])

    return _vectorize_point(*args_positions, manipulate_input=_manipulate_input)


def _vectorize_graph_to_points(*args_positions):
    r"""Vectorize GraphPoint or array into a list of GraphPoint.

    This way all the methods only need to work for the set of points case (output
    shape should be returned accordingly to the input though).

    Parameters
    ----------
    points : array-like, shape=[n_obs, n_nodes, n_nodes] or GraphPoint
        Adjacency matrix or GraphPoint.

    Returns
    -------
    graph-set : list of GraphPoint points, shape=[n_obs, *]
        List of points GraphPoint.
    """

    def _manipulate_input(arg):
        if type(arg) in [list, tuple]:
            return arg

        if type(arg) is GraphPoint:
            return [arg]

        if arg.ndim == 2:
            return [GraphPoint(arg)]
        else:
            return [GraphPoint(point) for point in arg]

    return _vectorize_point(*args_positions, manipulate_input=_manipulate_input)


def _pad_with_zeros(*args_positions, copy=True):
    r"""Pad graphs with zeros."""

    def _dec(func):
        def _manipulate_input(points, n_nodes):
            return _pad_points_with_zeros(points, n_nodes, copy=copy)

        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            args = list(args)

            n_nodes = args[0].n_nodes
            for pos, name in args_positions:
                if name in kwargs:
                    kwargs[name] = _manipulate_input(kwargs[name], n_nodes)
                else:
                    args[pos] = _manipulate_input(args[pos], n_nodes)

            return func(*args, **kwargs)

        return _wrapped

    return _dec


class GraphPoint(Point):
    r"""Class for the GraphPoint.

    Points are represented by :math:`nodes \times nodes` adjacency matrices.

    Parameters
    ----------
    adj : array-like, shape=[n_nodes, n_nodes]
        Adjacency matrix.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(self, adj):
        super().__init__()
        self.adj = adj

    @property
    def n_nodes(self):
        """Retrieve the number of nodes."""
        return self.adj.shape[0]

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"Adjacency: {self.adj}"

    def __hash__(self):
        """Return the hash of the instance."""
        return hash(self.adj)

    def to_array(self):
        """Return a copy of the adjacency matrix."""
        return gs.copy(self.adj)

    def to_networkx(self):
        """Turn the graph into a networkx format."""
        return nx.from_numpy_array(self.adj)


class GraphSpace(PointSet):
    r"""Class for the Graph Space.

    Graph Space to analyse populations of labelled and unlabelled graphs.
    The space focuses on graphs with scalar euclidean attributes on nodes and edges,
    with a finite number of nodes and both directed and undirected edges.
    For undirected graphs, use symmetric adjacency matrices. The space is a quotient
    space obtained by applying the permutation action of nodes to the space
    of adjacency matrices. Notice that for computation reasons the module works with
    both the `gs.array` representation of graph and the `GraphPoint` representation.

    Points are represented by :math:`nodes \times nodes` adjacency matrices.
    Both the array input and the Graph Point type input work.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes
    total_space : space
        Total Space before applying the permutation action. Default: Euclidean Space.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    .. [Jain2009] Jain, B., Obermayer, K.
        "Structure Spaces." Journal of Machine Learning Research, 10(11), 2009.
        https://www.jmlr.org/papers/volume10/jain09a/jain09a.pdf
    """

    def __init__(self, n_nodes, total_space=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.total_space = (
            Matrices(n_nodes, n_nodes) if total_space is None else total_space
        )

    @_pad_with_zeros((1, "graphs"))
    def belongs(self, graphs, atol=gs.atol):
        r"""Check if the point belongs to the space.

        The adjacency matrix should be associated to the
        graph with n nodes.

        Parameters
        ----------
        graphs : list of GraphPoint or array-like, shape=[..., n, n].
                Points to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,n]
            Boolean denoting if graph belongs to the space.
        """
        if type(graphs) in [list, tuple]:
            return gs.array([graph.n_nodes == self.n_nodes for graph in graphs])
        elif type(graphs) is GraphPoint:
            return graphs.n_nodes == self.n_nodes

        return self.total_space.belongs(graphs, atol=atol)

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample in Graph Space.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

        Returns
        -------
        graph_samples : array-like, shape=[..., n, n]
            Points sampled in GraphSpace(n).
        """
        return self.total_space.random_point(n_samples=n_samples, bound=bound)

    @_vectorize_graph((1, "points"))
    @_pad_with_zeros((1, "points"))
    def set_to_array(self, points):
        r"""Return a copy of the adjacency matrices.

        Parameters
        ----------
        points : list of GraphPoint or array-like, shape=[..., n, n].
            Points to be turned into an array
        Returns
        -------
        graph_array : array-like, shape=[..., nodes, nodes]
            An array containing all the Graphs.
        """
        return gs.copy(points)

    @_vectorize_graph_to_points((1, "points"))
    @_pad_with_zeros((1, "points"))
    def set_to_networkx(self, points):
        r"""Turn points into a networkx object.

        Parameters
        ----------
        points : list of GraphPoint or array-like, shape=[..., n, n].

        Returns
        -------
        nx_list : list of Networkx object
            An array containing all the Graphs.
        """
        networkx_objs = [pt.to_networkx() for pt in points]
        return networkx_objs if len(networkx_objs) > 1 else networkx_objs[0]

    @_vectorize_graph((1, "graph_to_permute"))
    @_pad_with_zeros((1, "graph_to_permute"))
    def permute(self, graph_to_permute, permutation):
        r"""Permutation action applied to graph observation.

        Parameters
        ----------
        graph_to_permute : list of GraphPoint or array-like, shape=[..., n, n].
            Input graphs to be permuted.
        permutation: array-like, shape=[..., n]
            Node permutations where in position i we have the value j meaning
            the node i should be permuted with node j.

        Returns
        -------
        graphs_permuted : array-like, shape=[..., n, n]
            Graphs permuted.
        """

        def _get_permutation_matrix(indices_):
            return gs.array_from_sparse(
                data=gs.ones(self.n_nodes, dtype=gs.int64),
                indices=list(zip(range(self.n_nodes), indices_)),
                target_shape=(self.n_nodes, self.n_nodes),
            )

        if gs.ndim(permutation) == 1:
            perm_matrices = _get_permutation_matrix(permutation)
        else:
            perm_matrices = []
            for indices_ in permutation:
                perm_matrices.append(_get_permutation_matrix(indices_))
            perm_matrices = gs.stack(perm_matrices)

        permuted_graph = Matrices.mul(
            perm_matrices, graph_to_permute, Matrices.transpose(perm_matrices)
        )
        if gs.ndim(permuted_graph) == 3 and gs.shape(permuted_graph)[0] == 1:
            return permuted_graph[0]

        return permuted_graph

    @_pad_with_zeros((1, "points"), copy=False)
    def pad_with_zeros(self, points):
        """Pad points with zeros to match space dimension.

        Parameters
        ----------
        points : list of GraphPoint or array-like, shape=[..., n, n].
        """
        return points


class GraphSpaceMetric(PointSetMetric):
    r"""Class for the Graph Space Metric.

    Every metric :math:`d: X \times X \rightarrow \mathbb{R}` on the total space of
    adjacency matrices can descend onto the quotient space as a pseudo-metric:
    :math:`d([x_1],[x_2]) = min_{t\in T} d_X(x_1, t^Tx_2t)`. The metric relies on the
    total space metric and an alignment procedure, i.e., Graph Matching or Networks
    alignment procedure. Metric, alignment, geodesics, and alignment with respect to
    a geodesic are defined. By default, the alignment is the identity and the total
    space metric is the Frobenious norm.

    Parameters
    ----------
    space : GraphSpace

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(self, space):
        super().__init__(space)
        self.aligner = self._set_default_aligner()
        self.point_to_geodesic_aligner = None

    @property
    def perm_(self):
        r"""Permutation of nodes after alignment.

        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
        """
        return self.aligner.perm_

    def _set_default_aligner(self):
        return self.set_aligner("ID")

    def set_aligner(self, aligner, **kwargs):
        r"""Set the aligning strategy.

        Graph Space metric relies on alignment. In this module we propose the
        identity matching, the FAQ graph matching by [Vogelstein2015], and
        exhaustive aligner which explores the whole permutation group.

        Parameters
        ----------
        aligner : str
            'ID' Identity
            'FAQ' Fast Quadratic Assignment - only compatible with Frobenious norm
            'exhaustive' all group exhaustive search

        References
        ----------
        .. [Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
            Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
            “Fast approximate quadratic programming for graph matching.“
            PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
        """
        if isinstance(aligner, str):
            MAP_ALIGNER = {
                "ID": IDAligner,
                "FAQ": FAQAligner,
                "exhaustive": ExhaustiveAligner,
            }
            check_parameter_accepted_values(
                aligner, "aligner", list(MAP_ALIGNER.keys())
            )

            aligner = MAP_ALIGNER.get(aligner)(**kwargs)

        self.aligner = aligner
        return self.aligner

    def set_point_to_geodesic_aligner(self, aligner, **kwargs):
        r"""Set the alignment between a point and a geodesic.

        Following the geodesic to point alignment in [Calissano2020] and
        [Huckemann2010], this function define the parameters [s_min, s_max] and
        the number of points to sample in the domain.

        Parameters
        ----------
        s_min : float
        s_max : float
        n_points: int

        References
        ----------
        .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
            “Graph Space: Geodesic Principal Components for a Population of
            Network-valued Data.” Mox report 14, 2020.
            https://mox.polimi.it/reports-and-theses/publication-results/?id=855.

        .. [Huckemann2010] Huckemann, S., Hotz, T., Munk, A.
            "Intrinsic shape analysis: Geodesic PCA for Riemannian manifolds modulo
            isometric Lie group actions." Statistica Sinica, 1-58, 2010.
        """
        if aligner == "default":
            kwargs.setdefault("s_min", -1.0)
            kwargs.setdefault("s_max", 1.0)
            kwargs.setdefault("n_points", 10)
            aligner = PointToGeodesicAligner(metric=self, **kwargs)

        self.point_to_geodesic_aligner = aligner
        return self.point_to_geodesic_aligner

    @property
    def total_space_metric(self):
        """Retrieve the total space metric."""
        return self.space.total_space.metric

    @total_space_metric.setter
    def total_space_metric(self, value):
        """Set the total space metric."""
        self.space.total_space.metric = value

    @property
    def n_nodes(self):
        """Retrieve the number of nodes."""
        return self.space.n_nodes

    @_vectorize_graph((1, "graph_a"), (2, "graph_b"))
    @_pad_with_zeros((1, "graph_a"), (2, "graph_b"))
    def dist(self, graph_a, graph_b):
        """Compute distance between two equivalence classes.

        Compute the distance between two equivalence classes of
        adjacency matrices [Jain2009].

        Parameters
        ----------
        graph_a : list of GraphPoint or array-like, shape=[..., n, n].
        graph_b : list of GraphPoint or array-like, shape=[..., n, n].

        Returns
        -------
        distance : array-like, shape=[...]
            distance between equivalence classes.

        References
        ----------
        .. [Jain2009]  Jain, B., Obermayer, K.
            "Structure Spaces." Journal of Machine Learning Research 10.11 (2009).
            https://www.jmlr.org/papers/v10/jain09a.html.
        """
        aligned_graph_b = self.align_point_to_point(graph_a, graph_b)
        return self.total_space_metric.dist(
            graph_a,
            aligned_graph_b,
        )

    @_vectorize_graph((1, "base_point"), (2, "end_point"))
    @_pad_with_zeros((1, "base_point"), (2, "end_point"))
    def geodesic(self, base_point, end_point):
        """Compute geodesic between two equivalence classes.

        Compute the geodesic between two equivalence classes of
        adjacency matrices [Calissano2020].

        Parameters
        ----------
        base_point : list of GraphPoint or array-like, shape=[..., n, n].
            Start .
        end_point : list of GraphPoint or array-like, shape=[..., n, n].
            Second graph to align to the first graph.

        Returns
        -------
        geodesic : function
            geodesic function.

        References
        ----------
        .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
        """
        aligned_end_point = self.align_point_to_point(base_point, end_point)

        return self.total_space_metric.geodesic(
            initial_point=base_point, end_point=aligned_end_point
        )

    @_vectorize_graph((1, "base_graph"), (2, "graph_to_permute"))
    @_pad_with_zeros((1, "base_graph"), (2, "graph_to_permute"))
    def align_point_to_point(self, base_graph, graph_to_permute):
        """Align graphs.

        Using the selected alignment technique, it returns the permuted
        graph_to_permute as optimally aligned to the base_graph.

        Parameters
        ----------
        base_graph : list of Graph or array-like, shape=[..., n, n].
            Base graph.
        graph_to_permute : list of Graph or array-like, shape=[..., n, n].
            Graph to align.

        Returns
        -------
        permuted_graph: list, shape = [...,n, n]
        """
        return self.aligner.align(self, base_graph, graph_to_permute)

    @_vectorize_graph(
        (2, "graph_to_permute"),
    )
    @_pad_with_zeros(
        (2, "graph_to_permute"),
    )
    def align_point_to_geodesic(self, geodesic, graph_to_permute):
        """Align graph to a geodesic.

        Using the selected alignment technique, it returns the permuted
        graph_to_permute as optimally aligned to the geodesic using [Huckemann2010].

        Parameters
        ----------
        geodesic : function.

        graph_to_permute : list of Graph or array-like, shape=[..., n, n].
            Graph to align.

        Returns
        -------
        permuted_graph: list, shape = [...,n, n]

        References
        ----------
        .. [Huckemann2010] Huckemann, S., Hotz, T., Munk, A.
            "Intrinsic shape analysis: Geodesic PCA for Riemannian manifolds modulo
            isometric Lie group actions." Statistica Sinica, 1-58, 2010.
        """
        if self.point_to_geodesic_aligner is None:
            raise UnboundLocalError(
                "Set point to geodesic aligner first (e.g. "
                "`metric.set_point_to_geodesic_aligner('default', "
                "s_min=-1., s_max=1.)`)"
            )
        return self.point_to_geodesic_aligner.align(geodesic, graph_to_permute)


class _BaseAligner(metaclass=ABCMeta):
    """Base class for point to point aligner.

    Attributes
    ----------
    perm_ : array-like, shape=[...,n]
        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
    """

    def __init__(self):
        self.perm_ = None

    @abstractmethod
    def align(self, metric, base_graph, graph_to_permute):
        """Align graphs.

        Parameters
        ----------
        metric : GraphSpaceMetric
        base_graph : array-like, shape=[..., n_nodes, n_nodes]
            Base graph.
        graph_to_permute : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        permuted_graph : array-like, shape=[..., n_nodes, n_nodes]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        raise NotImplementedError("Not implemented")

    def _broadcast(self, base_graph, graph_to_permute):
        base_graph, graph_to_permute = gs.broadcast_arrays(base_graph, graph_to_permute)
        is_single = gs.ndim(base_graph) == 2
        if is_single:
            base_graph = gs.expand_dims(base_graph, 0)
            graph_to_permute = gs.expand_dims(graph_to_permute, 0)

        return base_graph, graph_to_permute, is_single

    def _permute(self, metric, graph_to_permute, perm):
        return metric.space.permute(graph_to_permute, perm)


class FAQAligner(_BaseAligner):
    """Fast Quadratic Assignment for graph matching (or network alignment).

    References
    ----------
    .. [Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
        Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
        “Fast approximate quadratic programming for graph matching.“
        PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
    """

    def align(self, metric, base_graph, graph_to_permute):
        """Align graphs.

        Parameters
        ----------
        metric : GraphSpaceMetric
        base_graph : array-like, shape=[..., n_nodes, n_nodes]
            Base graph.
        graph_to_permute : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        permuted_graph : array-like, shape=[..., n_nodes, n_nodes]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        base_graph, graph_to_permute, is_single = self._broadcast(
            base_graph, graph_to_permute
        )

        perm = [
            gs.linalg.quadratic_assignment(x, y, options={"maximize": True})
            for x, y in zip(base_graph, graph_to_permute)
        ]

        self.perm_ = gs.array(perm[0]) if is_single else gs.array(perm)

        return self._permute(metric, graph_to_permute, self.perm_)


class IDAligner(_BaseAligner):
    """Identity alignment.

    The identity alignment is not performing any matching but returning the nodes in
    their original position. This alignment can be selected when working with labelled
    graphs.
    """

    def align(self, metric, base_graph, graph_to_permute):
        """Align graphs.

        Parameters
        ----------
        metric : GraphSpaceMetric
        base_graph : array-like, shape=[..., n_nodes, n_nodes]
            Base graph.
        graph_to_permute : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        permuted_graph : array-like, shape=[..., n_nodes, n_nodes]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        base_graph, graph_to_permute, is_single = self._broadcast(
            base_graph, graph_to_permute
        )

        n_nodes = base_graph.shape[1]
        perm = gs.reshape(gs.tile(range(n_nodes), base_graph.shape[0]), (-1, n_nodes))

        self.perm_ = gs.array(perm[0]) if is_single else gs.array(perm)

        return self._permute(metric, graph_to_permute, self.perm_)


class ExhaustiveAligner(_BaseAligner):
    """Brute force exact alignment.

    Exact Alignment obtained by exploring the whole permutation group.

    Notes
    -----
    Not recommended for large `n_nodes`.
    """

    def __init__(self):
        super().__init__()
        self._all_perms = None
        self._n_nodes = None

    def _set_all_perms(self, metric):
        n_nodes = metric.n_nodes
        if self._all_perms is None or self._n_nodes != n_nodes:
            self._n_nodes = n_nodes
            self._all_perms = gs.array(
                list(itertools.permutations(range(n_nodes), n_nodes))
            )

    def _align_single(self, metric, base_graph, graph_to_permute):
        permuted_graphs = metric.space.permute(graph_to_permute, self._all_perms)
        dists = metric.total_space_metric.dist(base_graph, permuted_graphs)
        return self._all_perms[gs.argmin(dists)]

    def align(self, metric, base_graph, graph_to_permute):
        """Align graphs.

        Parameters
        ----------
        metric : GraphSpaceMetric
        base_graph : array-like, shape=[..., n_nodes, n_nodes]
            Base graph.
        graph_to_permute : array-like, shape=[..., n_nodes, n_nodes]
            Graph to align.

        Returns
        -------
        permuted_graph : array-like, shape=[..., n_nodes, n_nodes]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        self._set_all_perms(metric)

        base_graph, graph_to_permute, is_single = self._broadcast(
            base_graph, graph_to_permute
        )
        perms = [
            self._align_single(metric, base_graph_, graph_to_permute_)
            for base_graph_, graph_to_permute_ in zip(base_graph, graph_to_permute)
        ]

        self.perm_ = gs.array(perms[0]) if is_single else gs.array(perms)

        return self._permute(metric, graph_to_permute, self.perm_)


class _BasePointToGeodesicAligner(metaclass=ABCMeta):
    """Base class for point to geodesic aligner.

    Attributes
    ----------
    perm_ : array-like, shape=[...,n]
        Node permutations where in position i we have the value j meaning
        the node i should be permuted with node j.
    """

    def __init__(self, metric):
        self.metric = metric
        self.perm_ = None

    @abstractmethod
    def align(self, geodesic, x):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def dist(self, geodesic, x):
        raise NotImplementedError("Not implemented")

    def _get_n_points(self, x):
        return 1 if gs.ndim(x) == 2 else gs.shape(x)[0]

    def _permute(self, graph_to_permute, perm):
        return self.metric.space.permute(graph_to_permute, perm)


class PointToGeodesicAligner(_BasePointToGeodesicAligner):
    r"""Class for the Alignment of the points with respect to a geodesic.

    Implementing the algorithm in [Huckemann2010] to select an optimal alignment to a
    point with respect to a geodesic. The algorithm sample discrete set of n_points
    along the geodesic between [s_min, s_max] and find the permutation that gets closer
    to the datapoints along the geodesic.

    Parameters
    ----------
    metric : GraphSpaceMetric
    s_min : float.
    s_max: float.
    n_points : int.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    .. [Huckemann2010] Huckemann, S., Hotz, T., Munk, A.
        "Intrinsic shape analysis: Geodesic PCA for Riemannian manifolds modulo
        isometric Lie group actions." Statistica Sinica, 1-58, 2010.
    """

    def __init__(self, metric, s_min, s_max, n_points=10):
        super().__init__(metric)
        self.s_min = s_min
        self.s_max = s_max
        self.n_points = n_points
        self._s = None

    def __setattr__(self, attr_name, value):
        r"""Set attributes."""
        if attr_name in ["s_min", "s_max", "n_points"]:
            self._s = None

        return object.__setattr__(self, attr_name, value)

    def _discretize_s(self):
        r"""Compute the domain distretization."""
        return gs.linspace(self.s_min, self.s_max, num=self.n_points)

    @property
    def s(self):
        r"""Save the domain distretization."""
        if self._s is None:
            self._s = self._discretize_s()

        return self._s

    def _get_gamma_s(self, geodesic):
        r"""Evaluate the geodesic in s."""
        return geodesic(self.s)

    def _compute_dists(self, geodesic, x):
        gamma_s = self._get_gamma_s(geodesic)

        n_points = self._get_n_points(x)
        if n_points > 1:
            gamma_s = gs.repeat(gamma_s, n_points, axis=0)
            rep_x = gs.concatenate([x for _ in range(self.n_points)])
        else:
            rep_x = x

        dists = gs.reshape(self.metric.dist(gamma_s, rep_x), (self.n_points, n_points))

        min_dists_idx = gs.argmin(dists, axis=0)

        return dists, min_dists_idx, n_points

    def dist(self, geodesic, graph_to_permute):
        r"""Compute the distance between the geodesic and the point.

        Parameters
        ----------
        geodesic : function.
        graph_to_permute : array-like, shape=[..., n, n]
            Graph to align.

        Returns
        -------
        dist : array-like, shape=[...,n]
            Distance between the graph_to_permute and the geodesic.
        """
        dists, min_dists_idx, n_points = self._compute_dists(geodesic, graph_to_permute)

        return gs.take(
            gs.transpose(dists),
            min_dists_idx + gs.arange(n_points) * self.n_points,
        )

    def align(self, geodesic, graph_to_permute):
        r"""Align the graph to the geodesic.

        Parameters
        ----------
        geodesic : function.
        graph_to_permute : array-like, shape=[..., n, n]
            Graph to align.

        Returns
        -------
        permuted_graph : array-like, shape=[...,n]
            Permuted graph as to be aligned with respect to the geodesic.
        """
        _, min_dists_idx, n_points = self._compute_dists(geodesic, graph_to_permute)

        perm_indices = min_dists_idx * n_points + gs.arange(n_points)
        if n_points == 1:
            perm_indices = perm_indices[0]

        self.perm_ = gs.take(self.metric.perm_, perm_indices, axis=0)

        return self._permute(graph_to_permute, self.perm_)


class _GeodesicToPointAligner(_BasePointToGeodesicAligner):
    def __init__(self, metric, method="BFGS", *, save_opt_res=False):
        super().__init__(metric)

        self.method = method
        self.save_opt_res = save_opt_res

        self.opt_results_ = None

    def _objective(self, s, x, geodesic):
        point = geodesic(s)
        dist = self.metric.dist(point, x)

        return dist

    def _compute_dists(self, geodesic, x):
        n_points = self._get_n_points(x)

        if n_points == 1:
            x = gs.expand_dims(x, axis=0)

        perms = []
        min_dists = []
        opt_results = []
        for xx in x:
            s0 = 0.0
            res = scipy.optimize.minimize(
                self._objective, x0=s0, args=(xx, geodesic), method=self.method
            )
            perms.append(self.metric.perm_[0])
            min_dists.append(res.fun)

            opt_results.append(res)

        if self.save_opt_res:
            self.opt_results_ = opt_results

        return gs.array(min_dists), gs.array(perms), n_points

    def dist(self, geodesic, x):
        dists, _, _ = self._compute_dists(geodesic, x)

        return dists

    def align(self, geodesic, x):
        _, perms, n_points = self._compute_dists(geodesic, x)

        new_x = self._permute(x, perms)
        self.perm_ = perms[0] if n_points == 1 else perms

        return new_x
