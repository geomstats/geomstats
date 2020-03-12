"""Prepare and Process Graph-structured data."""

import random


class Graph():
    """Class for generating a graph object from a dataset.

    Prepare Graph object from a dataset file.

    Parameters
    ----------
    Graph_Matrix_Path : string
        Path to graph adjacency matrix.

    Labels_Path : string
        Path to labels of the nodes of the graph.
    """

    edges = None
    labels = None
    paths = None
    walk_length = 5

    def __init__(self,
                 Graph_Matrix_Path=r'examples\data_example'
                                   r'\graph_random\Graph_Example_Random.txt',
                 Labels_Path=r'examples\data_example'
                             r'\graph_random\Graph_Example_Random_Labels.txt'):
        self.edges = {}
        with open(Graph_Matrix_Path, "r") as edges_file:
            for i, line in enumerate(edges_file):
                lsp = line.split()
                self.edges[i] = [k for k, value in
                                 enumerate(lsp) if (int(value) == 1)]

        if Labels_Path is not None:
            self.labels = {}
            with open(Labels_Path, "r") as label_file:
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
        self.walk_length = walk_length

        paths = []
        for index in range(len(self.edges)):
            for i in range(number_walks_per_node):
                paths.append(self._walk(index))

        return paths

    def _walk(self, index):
        """Generate a single random walk."""
        path = []
        c_index = index
        path.append(c_index)
        for i in range(self.walk_length):
            c_index = self.edges[c_index][random.randint(
                0, len(self.edges[c_index]) - 1)]
            path.append(c_index)
        return path
