"""Learning embeddings of data on manifolds."""

import logging

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall


class Embedding ():
    """A class for learning embeddings of data on manifolds.

    Currently supports embedding of graph-structured data on
    hyperbolic space.

    Parameters
    ----------
    data : object
        An instance of a data object.
    manifold : object
        An instance of a manifold object.
    dim : int
        Number of dimensions of the manifold.
    max_epochs : int
        Maximum number of iterations for embedding.
    lr : int
        Learning rate for embedding.
    n_negative : int
        Number of negative samples to consider.
    context_size : int
        Size of the context size to consider.
    """

    data = None
    manifold = None
    dim = None
    max_epochs = None
    lr = None
    n_negative = None
    context_size = None

    def __init__(
            self, data, manifold, dim, max_epochs,
            lr, n_negative, context_size,):

        self.data = data
        self.manifold = manifold
        self.dim = dim
        self.max_epochs = max_epochs
        self.lr = lr
        self.n_negative = n_negative
        self.context_size = context_size

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

    @staticmethod
    def grad_squared_distance(point_a, point_b):
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
        hyperbolic_metric = PoincareBall(2).metric
        log_map = hyperbolic_metric.log(point_b, point_a)

        return -2 * log_map

    def loss(self, example_embedding, context_embedding, negative_embedding,
             ):
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

        total_loss = -(positive_loss + negative_loss.sum())

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

        example_grad = -(positive_grad + negative_grad.sum(axis=0))

        return total_loss, example_grad

    def embed(self):
        """Compute embedding.

        Optimize a loss function to obtain a representable embedding.

        Returns
        -------
        embeddings : array-like, shape=[n_samples, dim]
            Return the embedding of the data. Each data sample
            is represented as a point belonging to the manifold.
        """
        nb_vertices_by_edges = \
            [len(e_2) for _, e_2 in self.data.edges.items()]
        logging.info('Number of edges: %s', len(self.data.edges))
        logging.info(
            'Mean vertices by edges: %s',
            (sum(nb_vertices_by_edges, 0) / len(self.data.edges)))

        negative_table_parameter = 5
        negative_sampling_table = []

        for i, nb_v in enumerate(nb_vertices_by_edges):
            negative_sampling_table += \
                ([i] * int((nb_v ** (3. / 4.))) * negative_table_parameter)

        negative_sampling_table = gs.array(negative_sampling_table)
        random_walks = self.data.random_walk()
        embeddings = gs.random.normal(size=(self.data.n_nodes, self.dim))
        embeddings = embeddings * 0.2

        for epoch in range(self.max_epochs):
            total_loss = []
            for path in random_walks:

                for example_index, one_path in enumerate(path):
                    context_index = path[max(
                        0, example_index - self.context_size):
                            min(example_index + self.context_size,
                                len(path))]
                    negative_index = \
                        gs.random.randint(negative_sampling_table.shape[0],
                                          size=(len(context_index),
                                                self.n_negative))
                    negative_index = negative_sampling_table[negative_index]

                    example_embedding = embeddings[one_path]

                    for one_context_i, one_negative_i in zip(context_index,
                                                             negative_index):
                        context_embedding = embeddings[one_context_i]
                        negative_embedding = embeddings[one_negative_i]
                        l, g_ex = self.loss(
                            example_embedding,
                            context_embedding,
                            negative_embedding)
                        total_loss.append(l)

                        example_to_update = embeddings[one_path]
                        embeddings[one_path] = self.manifold.metric.exp(
                            -self.lr * g_ex, example_to_update)

            logging.info(
                'iteration %d loss_value %f',
                epoch, sum(total_loss, 0) / len(total_loss))
        return embeddings
