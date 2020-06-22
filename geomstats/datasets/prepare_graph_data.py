"""Prepare and process graph-structured data."""

import logging
import random

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall


class Graph:
    """Class for generating a graph object from a dataset.

    Prepare Graph object from a dataset file.

    Parameters
    ----------
    graph_matrix_path : string
        Path to graph adjacency matrix.
    labels_path : string
        Path to labels of the nodes of the graph.

    Attributes
    ----------
    edges : dict
        Dictionary with node number as key
        and edge connected node numbers as values.
    n_nodes : int
        Number of nodes in the graph.
    labels : dict
        Dictionary with node number as key and the true label number as values.
    """

    def __init__(self, graph_matrix_path, labels_path):
        self.edges = {}
        with open(graph_matrix_path, 'r') as edges_file:
            for i, line in enumerate(edges_file):
                lsp = line.split()
                self.edges[i] = [k for k, value in
                                 enumerate(lsp) if int(value) == 1]

        self.n_nodes = len(self.edges)

        if labels_path is not None:
            self.labels = {}
            with open(labels_path, 'r') as labels_file:
                for i, line in enumerate(labels_file):
                    self.labels[i] = []
                    self.labels[i].append(int(line))

    def random_walk(self, walk_length=5, n_walks_per_node=1):
        """Compute a set of random walks on a graph.

        For each node of the graph, generates a a number of
        random walks of a specified length.
        Two consecutive nodes in the random walk, are necessarily
        related with an edge. The walks capture the structure of the graph.

        Parameters
        ----------
        walk_length : int
            Length of a random walk in terms of number of edges.
        n_walks_per_node : int
            Number of generated walks starting from each node of the graph.

        Returns
        -------
        self : array-like,
            Shape=[n_walks_per_node*self.n_edges), walk_length]
            array containing random walks.
        """
        paths = [[0] * (walk_length + 1)
                 for i in range(self.n_nodes * n_walks_per_node)]

        for index in range(len(self.edges)):
            for i in range(n_walks_per_node):
                paths[index * n_walks_per_node + i] =\
                    self._walk(index, walk_length)
        return gs.array(paths)

    def _walk(self, index, walk_length):
        """Generate a single random walk."""
        count_index = index
        path = [index]
        for _ in range(walk_length):
            count_index = self.edges[count_index][random.randint(
                0, len(self.edges[count_index]) - 1)]
            path.append(count_index)
        return gs.array(path, dtype=gs.int32)


class HyperbolicEmbedding:
    """Class for learning embeddings of graphs on hyperbolic space.

    Parameters
    ----------
    dim : object
        Dimensions of the used hyperbolic space.
    max_epochs : int
        Maximum number of iterations for embedding.
    lr : int
        Learning rate for embedding.
    n_context : int
        Number of nodes to consider from a neighborhood
        of nodes around a particular node.
    n_negative : int
        Number of nodes to consider when searching for
        a set of nodes that are far from a particular node.
    """

    def __init__(
            self, dim=2, max_epochs=100,
            lr=.05, n_context=1, n_negative=2):

        self.manifold = PoincareBall(dim)
        self.max_epochs = max_epochs
        self.lr = lr
        self.n_context = n_context
        self.n_negative = n_negative

    @staticmethod
    def log_sigmoid(vector):
        """Logsigmoid function.

        Apply log sigmoid function.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dim]

        Returns
        -------
        result : array-like, shape=[n_samples, dim]
        """
        return gs.log((1 / (1 + gs.exp(-vector))))

    @staticmethod
    def grad_log_sigmoid(vector):
        """Gradient of log sigmoid function.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dim]

        Returns
        -------
        gradient : array-like, shape=[n_samples, dim]
        """
        return 1 / (1 + gs.exp(vector))

    def grad_squared_distance(self, point_a, point_b):
        """Gradient of squared hyperbolic distance.

        Gradient of the squared distance based on the
        Ball representation according to point_a.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dim]
            First point in hyperbolic space.
        point_b : array-like, shape=[n_samples, dim]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
            Geodesic squared distance between the two points.
        """
        hyperbolic_metric = self.manifold.metric
        log_map = hyperbolic_metric.log(point_b, point_a)

        return -2 * log_map

    def loss(
            self, example_embedding, context_embedding, negative_embedding):
        """Compute loss and grad.

        Compute loss and grad given embedding of the current example,
        embedding of the context and negative sampling embedding.

        Parameters
        ----------
        example_embedding : array-like, shape=[dim]
            Current data sample embedding.
        context_embedding : array-like, shape=[dim]
            Current context embedding.
        negative_embedding: array-like, shape=[dim]
            Current negative sample embedding.

        Returns
        -------
        total_loss : int
            The current value of the loss function.
        example_grad : array-like, shape=[dim]
            The gradient of the loss function at the embedding
            of the current data sample.
        """
        n_edges, dim =\
            negative_embedding.shape[0], example_embedding.shape[-1]
        example_embedding = gs.expand_dims(example_embedding, 0)
        context_embedding = gs.expand_dims(context_embedding, 0)

        positive_distance =\
            self.manifold.metric.squared_dist(
                example_embedding, context_embedding)
        positive_loss =\
            self.log_sigmoid(-positive_distance)

        reshaped_example_embedding =\
            gs.repeat(example_embedding, n_edges, axis=0)

        negative_distance =\
            self.manifold.metric.squared_dist(
                reshaped_example_embedding, negative_embedding)
        negative_loss = self.log_sigmoid(negative_distance)

        total_loss = -(positive_loss + gs.sum(negative_loss))

        positive_log_sigmoid_grad =\
            -self.grad_log_sigmoid(-positive_distance)

        positive_distance_grad =\
            self.grad_squared_distance(example_embedding, context_embedding)

        positive_grad =\
            gs.repeat(positive_log_sigmoid_grad, dim, axis=-1)\
            * positive_distance_grad

        negative_distance_grad =\
            self.grad_squared_distance(
                reshaped_example_embedding, negative_embedding)

        negative_distance = gs.to_ndarray(negative_distance,
                                          to_ndim=2, axis=-1)
        negative_log_sigmoid_grad =\
            self.grad_log_sigmoid(negative_distance)

        negative_grad = negative_log_sigmoid_grad\
            * negative_distance_grad

        example_grad = -(positive_grad + gs.sum(negative_grad, axis=0))

        return total_loss, example_grad

    def embed(self, graph):
        """Compute embedding.

        Optimize a loss function to obtain a representable embedding.

        Parameters
        ----------
        graph : object
            An instance of the Graph class.

        Returns
        -------
        embeddings : array-like, shape=[n_samples, dim]
            Return the embedding of the data. Each data sample
            is represented as a point belonging to the manifold.
        """
        nb_vertices_by_edges = \
            [len(e_2) for _, e_2 in graph.edges.items()]
        logging.info('Number of edges: %s', len(graph.edges))
        logging.info(
            'Mean vertices by edges: %s',
            (sum(nb_vertices_by_edges, 0) / len(graph.edges)))

        negative_table_parameter = 5
        negative_sampling_table = []

        for i, nb_v in enumerate(nb_vertices_by_edges):
            negative_sampling_table += \
                ([i] * int((nb_v ** (3. / 4.))) * negative_table_parameter)

        negative_sampling_table = gs.array(negative_sampling_table)
        random_walks = graph.random_walk()
        embeddings = gs.random.normal(
            size=(graph.n_nodes, self.manifold.dim))
        embeddings = embeddings * 0.2

        for epoch in range(self.max_epochs):
            total_loss = []
            for path in random_walks:

                for example_index, one_path in enumerate(path):
                    context_index = path[max(
                        0, example_index - self.n_context):
                            min(example_index + self.n_context,
                                len(path))]
                    negative_index = \
                        gs.random.randint(negative_sampling_table.shape[0],
                                          size=(len(context_index),
                                                self.n_negative))

                    negative_index = gs.expand_dims(
                        gs.flatten(negative_index), axis=-1)

                    negative_index = gs.get_slice(
                        negative_sampling_table, negative_index)

                    example_embedding = embeddings[gs.cast(
                        one_path, dtype=gs.int64)]

                    for one_context_i, one_negative_i in zip(context_index,
                                                             negative_index):
                        context_embedding = embeddings[one_context_i]

                        negative_embedding = gs.get_slice(
                            embeddings, gs.squeeze(gs.cast(
                                one_negative_i, dtype=gs.int64)))

                        l, g_ex = self.loss(
                            example_embedding,
                            context_embedding,
                            negative_embedding)
                        total_loss.append(l)

                        example_to_update = embeddings[one_path]

                        valeur = self.manifold.metric.exp(
                            -self.lr * g_ex, example_to_update)

                        embeddings = gs.assignment(
                            embeddings, valeur, gs.to_ndarray(
                                one_path, to_ndim=1), axis=1)

            logging.info(
                'iteration %d loss_value %f',
                epoch, sum(total_loss, 0) / len(total_loss))
        return embeddings
