"""Graph Space.

Lead author: Anna Calissano.
"""

import networkx as nx

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stratified.point_set import (
    Point,
    PointSet,
    PointSetMetric,
    _vectorize_point,
)


def _manipulate_input(arg):
    if type(arg) not in [list, tuple, Graph]:
        return arg

    if type(arg) is Graph:
        return arg.adj

    return gs.array([graph.adj for graph in arg])


def _vectorize_graph(*args_positions):
    return _vectorize_point(*args_positions, manipulate_input=_manipulate_input)


class Graph(Point):
    r"""Class for the Graph.

    Points are represented by :math:`nodes \times nodes` adjacency matrices.

    Parameters
    ----------
    adj : array-like, shape=[n_nodes, n_nodes]
        Adjacency matrix.

    References
    ----------
    ..[Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
              “Graph Space: Geodesic Principal Components for a Population of
              Network-valued Data.”
              Mox report 14, 2020.
              https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(self, adj):
        super(Graph).__init__()
        self.adj = adj

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"Adjacency: {self.adj}"

    def __hash__(self):
        """Return the hash of the instance."""
        return hash(self.adj)

    def to_array(self):
        """Return the hash of the instance."""
        return gs.copy(self.adj)

    def to_networkx(self):
        """Turn the graph into a networkx format."""
        return nx.from_numpy_matrix(self.adj)


class GraphSpace(PointSet):
    r"""Class for the Graph Space.

    Graph Space to analyse populations of labelled and unlabelled graphs.
    The space focuses on graphs with scalar euclidean attributes on nodes and edges,
    with a finite number of nodes and both directed and undirected edges.
    For undirected graphs, use symmetric adjacency matrices. The space is a quotient
    space obtained by applying the permutation action of nodes to the space
    of adjacency matrices. Notice that for computation reasons the module works with
    both the gs.array representation of graph and the Graph(Point) representation.

    Points are represented by :math:`nodes \times nodes` adjacency matrices.
    Both the array input and the Graph Point type input work.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes
    total_space : space
        Total Space before applying the permutation action. Default: Adjacency Matrices.

    References
    ----------
    ..[Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
              “Graph Space: Geodesic Principal Components for a Population of
              Network-valued Data.”
              Mox report 14, 2020.
              https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(self, n_nodes, total_space=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.total_space = (
            Matrices(n_nodes, n_nodes) if total_space is None else total_space
        )

    @_vectorize_graph((1, "graphs"))
    def belongs(self, graphs, atol=gs.atol):
        r"""Check if the matrix is an adjacency matrix.

        The adjacency matrix should be associated to the
        graph with n nodes.

        Parameters
        ----------
        graphs : list of Graph or array-like, shape=[..., n, n].
                Points to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,n]
            Boolean denoting if graph belongs to the space.
        """
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
    def set_to_array(self, points):
        r"""Sample in Graph Space.

        Parameters
        ----------
        points : list of Graph or array-like, shape=[..., n, n].
                Points to be turned into an array
        Returns
        -------
        graph_array : array-like, shape=[..., nodes, nodes]
                An array containing all the Graphs.
        """
        return points

    @_vectorize_graph((1, "points"))
    def to_networkx(self, points):
        r"""Turn point into a networkx object.

        Parameters
        ----------
        points : list of Graph or array-like, shape=[..., n, n].

        Returns
        -------
        nx_list : list of Networkx object
                An array containing all the Graphs.
        """
        if points.shape == (self.n_nodes, self.n_nodes):
            return nx.from_numpy_matrix(points)
        return [nx.from_numpy_matrix(point) for point in points]

    @_vectorize_graph((1, "graph_to_permute"))
    def permute(self, graph_to_permute, permutation):
        r"""Permutation action applied to graph observation.

        Parameters
        ----------
        graph_to_permute : list of Graph or array-like, shape=[..., n, n].
            Input graphs to be permuted.
        permutation: array-like, shape=[..., n]
            Node permutations where in position i we have the value j meaning
            the node i should be permuted with node j.

        Returns
        -------
        graphs_permuted : array-like, shape=[..., n, n]
            Graphs permuted.
        """
        nodes = self.n_nodes
        single_graph = len(graph_to_permute.shape) < 3
        if single_graph:
            graph_to_permute = [graph_to_permute]
            permutation = [permutation]
        result = []
        for i, p in enumerate(permutation):
            if gs.all(gs.array(nodes) == gs.array(p)):
                result.append(graph_to_permute[i])
            else:
                gtype = graph_to_permute[i].dtype
                permutation_matrix = gs.array_from_sparse(
                    data=gs.ones(nodes, dtype=gtype),
                    indices=list(zip(list(range(nodes)), p)),
                    target_shape=(nodes, nodes),
                )
                result.append(
                    self.total_space.mul(
                        permutation_matrix,
                        graph_to_permute[i],
                        gs.transpose(permutation_matrix),
                    )
                )
        return result[0] if single_graph else gs.array(result)


class GraphSpaceMetric(PointSetMetric):
    """Quotient metric on the graph space.

    Parameters
    ----------
    space: Graph.
    """

    def __init__(self, space):
        self.space = space
        self.perm_ = None

    @property
    def total_space_metric(self):
        return self.space.total_space.metric

    @property
    def n_nodes(self):
        r"""Save the number of nodes."""
        return self.space.n_nodes

    @_vectorize_graph((1, "graph_a"), (2, "graph_b"))
    def dist(self, graph_a, graph_b, matcher="ID"):
        """Compute distance between two equivalence classes.

        Compute the distance between two equivalence classes of
        adjacency matrices [Jain2009]_.

        Parameters
        ----------
        graph_a : list of Graph or array-like, shape=[..., n, n].
        graph_b : list of Graph or array-like, shape=[..., n, n].
        matcher : str
            Check ``GraphSpace.matching``.

        Returns
        -------
        distance : array-like, shape=[...]
            distance between equivalence classes.

        References
        ----------
        ..[Jain2009]  Jain, B., Obermayer, K.
                  "Structure Spaces." Journal of Machine Learning Research 10.11 (2009).
                  https://www.jmlr.org/papers/v10/jain09a.html.
        """
        if graph_a.ndim > graph_b.ndim:
            base_graph = graph_b
            graph_to_permute = graph_a
        else:
            base_graph = graph_a
            graph_to_permute = graph_b

        # TODO: review above, otherwise self.perm_ loses meaning
        perm = self.matching(base_graph, graph_to_permute, matcher=matcher)

        return self.total_space_metric.dist(
            base_graph,
            self.space.permute(graph_to_permute, perm),
        )

    @_vectorize_graph((1, "base_point"), (2, "end_point"))
    def geodesic(self, base_point, end_point, matcher="ID"):
        """Compute distance between two equivalence classes.

        Compute the distance between two equivalence classes of
        adjacency matrices [Jain2009]_.

        Parameters
        ----------
        base_point : list of Graph or array-like, shape=[..., n, n].
            Start .
        end_point : list of Graph or array-like, shape=[..., n, n].
            Second graph to align to the first graph.
        matcher : str
            Check ``GraphSpace.matching``.

        Returns
        -------
        distance : array-like, shape=[...,]
            distance between equivalence classes.

        References
        ----------
        ..[Jain2009]  Jain, B., Obermayer, K.
                  "Structure Spaces." Journal of Machine Learning Research 10.11 (2009).
                  https://www.jmlr.org/papers/v10/jain09a.html.
        """
        perm = self.matching(base_point, end_point, matcher=matcher)

        return self.total_space_metric.geodesic(
            base_point, self.space.permute(end_point, perm)
        )

    @_vectorize_graph((1, "base_point"), (2, "end_point"))
    def matching(self, base_graph, graph_to_permute, matcher="ID"):
        """Match graphs.

        Parameters
        ----------
        base_graph : list of Graph or array-like, shape=[..., n, n].
            Base graph.
        graph_to_permute : list of Graph or array-like, shape=[..., n, n].
            Graph to align.
        matcher : str
            Possible values are 'ID', 'FAQ'.
                'ID': Identity matching.
                'FAQ': [Vogelstein2015]_ Fast Quadratic Assignment
            Note: uses Frobenius metric in background.

        References
        ----------
        ..[Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
                Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
                “Fast approximate quadratic programming for graph matching.“
                PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
        """
        matching_alg = {
            "ID": self._id_matching,
            "FAQ": self._faq_matching,
        }
        self.perm_ = matching_alg.get(matcher)(base_graph, graph_to_permute)

        return self.perm_

    @staticmethod
    @_vectorize_graph((0, "base_graph"), (1, "graph_to_permute"))
    def _faq_matching(base_graph, graph_to_permute):
        """Fast Quadratic Assignment for graph matching.

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
            Base graph.
        graph_to_permute : array-like, shape=[..., n, n]
            Graph to align.

        Returns
        -------
        permutation : array-like, shape=[...,n]
            node permutation indexes of the second graph.

        References
        ----------
        ..[Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
                Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
                “Fast approximate quadratic programming for graph matching.“
                PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
        """

        def _optimize(x, y):
            return gs.linalg.quadratic_assignment(x, y, options={"maximize": True})

        base_graph, graph_to_permute = gs.broadcast_arrays(base_graph, graph_to_permute)

        if gs.ndim(base_graph) == 1:
            return [_optimize(base_graph, graph_to_permute)]
        else:
            return [
                _optimize(base_graph_, graph_to_permute_)
                for base_graph_, graph_to_permute_ in zip(base_graph, graph_to_permute)
            ]

    @_vectorize_graph((1, "base_graph"), (2, "graph_to_permute"))
    def _id_matching(self, base_graph, graph_to_permute):
        """Identity matching.

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
            Base graph.
        graph_to_permute : array-like, shape=[..., n, n]
            Graph to align.

        Returns
        -------
        permutation : array-like, shape=[..., n]
            Node permutation indexes of the second graph.
        """
        base_graph, graph_to_permute = gs.broadcast_arrays(base_graph, graph_to_permute)

        n_reps = 1 if gs.ndim(base_graph) == 2 else base_graph.shape[0]
        if n_reps == 1:
            return gs.array(range(self.n_nodes))
        else:
            return gs.reshape(gs.tile(range(self.n_nodes), n_reps), (-1, self.n_nodes))
