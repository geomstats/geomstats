"""Prepare and Process Graph-structured data."""

import random

import geomstats.backend as gs


class Graph:
    """Class for generating a graph object from a dataset.

    Prepare Graph object from a dataset file.

    Parameters
    ----------
    graph_matrix_path : string
        Path to graph adjacency matrix.

    labels_path : string
        Path to labels of the nodes of the graph.
    """

    edges = None
    labels = None

    def __init__(self,
                 graph_matrix_path='examples\\data'
                                   '\\graph_random\\graph_random.txt',
                 labels_path='examples\\data\\graph_random'
                             '\\graph_random_labels.txt'):
        self.edges = {}
        with open(graph_matrix_path, 'r') as edges_file:
            for i, line in enumerate(edges_file):
                lsp = line.split()
                self.edges[i] = [k for k, value in
                                 enumerate(lsp) if (int(value) == 1)]

        if labels_path is not None:
            self.labels = {}
            with open(labels_path, 'r') as label_file:
                for i, line in enumerate(label_file):
                    self.labels[i] = []
                    self.labels[i].append(int(line))

    def random_walk(self, walk_length=5, number_walks_per_node=1):
        """Compute a set of random walks on a graph.

        For each node of the graph, generates a a number of
        random walks of a specified length.
        Two consecutive nodes in the random walk, are necessarily
        related with an edge. The walks capture the structure of the graph.

        Parameters
        ----------
        walk_length : int
            length of a random walk in terms of number of edges

        number_walks_per_node : int
            number of generated walks starting from each node of the graph

        Returns
        -------
        self : array-like,
            shape=[number_walks_per_node*len(self.edges), walk_length]
            array containing random walks
        """
        paths = gs.empty((0, walk_length + 1), dtype=int)
        for index in range(0, len(self.edges)):
            for i in range(number_walks_per_node):
                paths = gs.vstack((paths, self._walk(index, walk_length)))
        return paths

    def _walk(self, index, walk_length):
        """Generate a single random walk."""
        path = []
        count_index = index
        path = gs.append(path, count_index)
        for i in range(walk_length):
            count_index = self.edges[count_index][random.randint(
                0, len(self.edges[count_index]) - 1)]
            path = gs.append(path, count_index)
        return path
