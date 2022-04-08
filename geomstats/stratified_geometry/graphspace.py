"""Graph Space.

Lead author: Anna Calissano.
"""

import networkx as nx

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.stratified_geometry.stratified_spaces import (
    Point,
    PointSet,
    PointSetGeometry,
)


class Graph(Point):
    r"""Class for the Graph.

    Points are represented by :math:`nodes \times nodes` adjacency matrices.

    Parameters
    ----------
    nodes : int
        Number of graph nodes
    p : int
        Dimension of euclidean parameter or label associated to a graph.

    References
    ----------
    ..[Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
              “Graph Space: Geodesic Principal Components for a Population of
              Network-valued Data.”
              Mox report 14, 2020.
              https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(self, adj, p=None):
        super(Graph).__init__()
        self.adj = adj
        self.p = p

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"Adjacency: {self.adj} Parameter: {self.p}"

    def __hash__(self):
        """Return the hash of the instance."""
        return hash((self.adj, self.p))

    def to_array(self):
        """Return the hash of the instance."""
        return gs.array([self.adj, self.p])

    def to_networkx(self):
        """Turn the graph into a networkx format."""
        return nx.from_numpy_matrix(self.adj)


def graph_vectorize(fun):
    r"""Vectorize the input Graph Point to an array."""

    def wrapped(*args, **kwargs):
        r"""Vectorize the belongs."""
        args = list(args)
        if type(args[1]) not in [list, Graph]:
            return fun(*args, **kwargs)
        if type(args[1]) is Graph:
            args[1] = args[1].adj
            return fun(*args, **kwargs)
        args[1] = gs.array([graph.adj for graph in args[1]])
        return fun(*args, **kwargs)

    return wrapped


def multiple_input_vectorize(fun):
    r"""Vectorize the input Graph Point to an array."""

    def wrapped(*args, **kwargs):
        r"""Vectorize the belongs."""
        args = list(args)
        if type(args[1]) not in [list, Graph] and type(args[2]) not in [list, Graph]:
            return fun(*args, **kwargs)
        if type(args[1]) is Graph and type(args[2]) is Graph:
            args[1] = args[1].adj
            args[2] = args[2].adj
            return fun(*args, **kwargs)
        if type(args[1]) is Graph and type(args[2]) is list:
            args[1] = args[1].adj
            args[2] = gs.array([graph.adj for graph in args[2]])
            return fun(*args, **kwargs)
        args[1] = gs.array([graph.adj for graph in args[1]])
        args[2] = gs.array([graph.adj for graph in args[2]])
        return fun(*args, **kwargs)

    return wrapped


class GraphSpace(PointSet):
    r"""Class for the Graph Space.

    Graph Space to analyse populations of labelled and unlabelled graphs.
    The space focuses on graphs with scalar euclidean attributes on nodes and edges,
    with a finite number of nodes and both directed and undirected edges.
    For undirected graphs, use symmeric adjacency matrices. The space is a quotient
    space obtained by applying the permutation action of nodes to the space
    of adjacency matrices. Notice that for computation reasons the module works with
    both the gs.array representation of graph and the Graph(Point) representation.

    Points are represented by :math:`nodes \times nodes` adjacency matrices.

    Parameters
    ----------
    nodes : int
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

    def __init__(self, nodes, total_space=Matrices):
        super(GraphSpace).__init__()
        self.nodes = nodes
        self.total_space = total_space(self.nodes, self.nodes)

    @graph_vectorize
    def belongs(self, graph, atol=gs.atol):
        r"""Check if the matrix is an adjacency matrix.

        The adjacency matrix should be associated to the
        graph with n nodes.

        Parameters
        ----------
        points : List of Graph type or array-like Adjecency Matrices.
                Points to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,n]
            Boolean denoting if graph belongs to the space.
        """
        return self.total_space.belongs(graph, atol=atol)

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

    @graph_vectorize
    def set_to_array(self, points):
        r"""Sample in Graph Space.

        Parameters
        ----------
        points : List of Graph type or array-like Adjacency Matrices.
                Points to be turned into an array
        Returns
        -------
        graph_array : array-like, shape=[..., nodes, nodes]
                An array containing all the Graphs.
        """
        return points

    @graph_vectorize
    def to_networkx(self, points):
        r"""Turn point into a networkx object.

        Parameters
        ----------
        points : List of Graph type or array-like Adjacency Matrices.

        Returns
        -------
        nx_list : list of Networkx object
                An array containing all the Graphs.
        """
        if points.shape == (self.nodes, self.nodes):
            return nx.from_numpy_matrix(points)
        return [nx.from_numpy_matrix(point) for point in points]

    @graph_vectorize
    def permute(self, graph_to_permute, permutation):
        r"""Permutation action applied to graph observation.

        Parameters
        ----------
        graph_to_permute : array-like, shape=[..., n, n]
            Input graphs to be permuted.
        permutation: array-like, shape=[..., n]
            Node permutations where in position i we have the value j meaning
            the node i should be permuted with node j.

        Returns
        -------
        graphs_permuted : array-like, shape=[..., n, n]
            Graphs permuted.
        """
        nodes = self.nodes
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


class GraphSpaceMetric(PointSetGeometry):
    """Quotient metric on the graph space.

    Parameters
    ----------
    nodes : int
        Number of nodes
    """

    def __init__(self, nodes, total_space_metric=MatricesMetric, space=GraphSpace):
        self.nodes = nodes
        self.total_space_metric = total_space_metric(self.nodes, self.nodes)
        self.space = space(self.nodes)

    @multiple_input_vectorize
    def dist(self, base_graph, graph_to_permute, matcher="ID"):
        """Compute distance between two equivalence classes.

        Compute the distance between two equivalence classes of
        adjacency matrices [Jain2009]_.

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
            First graph.
        graph_to_permute : array-like, shape=[..., n, n]
            Second graph to align to the first graph.
        matcher : selecting which matcher to use
            'FAQ': [Vogelstein2015]_ Fast Quadratic Assignment
            note: use Frobenius metric in background.

        Returns
        -------
        distance : array-like, shape=[...,]
            distance between equivalence classes.

        References
        ----------
        ..[Jain2009]  Jain, B., Obermayer, K.
                  "Structure Spaces." Journal of Machine Learning Research 10.11 (2009).
                  https://www.jmlr.org/papers/v10/jain09a.html.
        ..[Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
                Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
                “Fast approximate quadratic programming for graph matching.“
                PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
        """
        if matcher == "FAQ":
            perm = self.faq_matching(base_graph, graph_to_permute)
        if matcher == "ID":
            perm = self.id_matching(base_graph, graph_to_permute)
        return self.total_space_metric.dist(
            base_graph,
            self.space.permute(graph_to_permute, perm),
        )

    @multiple_input_vectorize
    def geodesic(self, base_point, end_point, matcher="ID"):
        """Compute distance between two equivalence classes.

        Compute the distance between two equivalence classes of
        adjacency matrices [Jain2009]_.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]
            Start .
        point_b : array-like, shape=[..., n, n]
            Second graph to align to the first graph.
        matcher : selecting which matcher to use
            'FAQ': [Vogelstein2015]_ Fast Quadratic Assignment
            note: use Frobenius metric in background.

        Returns
        -------
        distance : array-like, shape=[...,]
            distance between equivalence classes.

        References
        ----------
        ..[Jain2009]  Jain, B., Obermayer, K.
                  "Structure Spaces." Journal of Machine Learning Research 10.11 (2009).
                  https://www.jmlr.org/papers/v10/jain09a.html.
        ..[Vogelstein2015] Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ,
                Kratzer SG, Harley ET, Fishkind DE, Vogelstein RJ, Priebe CE.
                “Fast approximate quadratic programming for graph matching.“
                PLoS One. 2015 Apr 17; doi: 10.1371/journal.pone.0121002.
        """
        return 0

    @staticmethod
    def faq_matching(base_graph, graph_to_permute):
        """Fast Quadratic Assignment for graph matching.

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
        First graph.
        graph_to_permute : array-like, shape=[..., n, n]
        Second graph to align.

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
        l_base = len(base_graph.shape)
        l_obj = len(graph_to_permute.shape)
        if l_base == l_obj == 3:
            return [
                gs.linalg.quadratic_assignment(x, y, options={"maximize": True})
                for x, y in zip(base_graph, graph_to_permute)
            ]
        if l_base == l_obj == 2:
            return gs.linalg.quadratic_assignment(
                base_graph, graph_to_permute, options={"maximize": True}
            )
        if l_base < l_obj:
            return [
                gs.linalg.quadratic_assignment(x, y, options={"maximize": True})
                for x, y in zip(
                    gs.stack([base_graph] * graph_to_permute.shape[0]), graph_to_permute
                )
            ]
        raise ValueError(
            "The method can align a set of graphs to one graphs,"
            "but the single graphs should be passed as base_graph"
        )

    def id_matching(self, base_graph, graph_to_permute):
        """Identity matching.

        Parameters
        ----------
        base_graph : array-like, shape=[..., n, n]
        First graph.
        graph_to_permute : array-like, shape=[..., n, n]
        Second graph to align.

        Returns
        -------
        permutation : array-like, shape=[...,n]
            node permutation indexes of the second graph.
        """
        l_base = len(base_graph.shape)
        l_obj = len(graph_to_permute.shape)
        if l_base == l_obj == 3 or l_base < l_obj:
            return [list(range(self.nodes))] * len(graph_to_permute)
        if l_base == l_obj == 2:
            return list(range(self.nodes))
        raise (
            ValueError(
                "The method can align a set of graphs to one graphs,"
                "but the single graphs should be passed as base_graph"
            )
        )
