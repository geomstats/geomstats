"""Learning embedding of graph using Poincare Ball Model."""

import logging

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.datasets import graph_data_preparation as gdp
from geomstats.geometry.hyperbolic import Hyperbolic


def log_sigmoid(vector):
    """Logsigmoid function.

    Apply log sigmoid function

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dimension]

    Returns
    -------
    result : array-like, shape=[n_samples, dimension]
    """
    return gs.log((1 / (1 + gs.exp(-vector))))


def grad_log_sigmoid(vector):
    """Gradient of log sigmoid function.

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dimension]

    Returns
    -------
    gradient : array-like, shape=[n_samples, dimension]
    """
    return (1 / (1 + gs.exp(vector)))


def grad_squared_distance(point_a, point_b):
    """Gradient of squared hyperbolic distance.

    Gradient of the squared distance based on the
    Ball representation according to point_a

    Parameters
    ----------
    point_a : array-like, shape=[n_samples, dimension]
        First point in hyperbolic space.
    point_b : array-like, shape=[n_samples, dimension]
        Second point in hyperbolic space.

    Returns
    -------
    dist : array-like, shape=[n_samples, 1]
        Geodesic squared distance between the two points.
    """
    hyperbolic_metric = Hyperbolic(2, coords_type='ball').metric
    log_map =\
        hyperbolic_metric.log(point_b, point_a)

    return -2 * log_map


def loss(example_embedding, context_embedding, negative_embedding,
         manifold):
    """Compute loss and grad.

    Compute loss and grad given embedding of the current example,
    embedding of the context and negative sampling embedding.
    """
    n_edges, dim =\
        negative_embedding.shape[0], example_embedding.shape[-1]
    example_embedding = gs.expand_dims(example_embedding, 0)
    context_embedding = gs.expand_dims(context_embedding, 0)

    positive_distance =\
        manifold.metric.squared_dist(example_embedding, context_embedding)

    positive_loss =\
        log_sigmoid(-positive_distance)

    reshaped_example_embedding =\
        gs.repeat(example_embedding, n_edges, axis=0)

    negative_distance =\
        manifold.metric.squared_dist(reshaped_example_embedding,
                                     negative_embedding)
    negative_loss = log_sigmoid(negative_distance)

    total_loss = -(positive_loss + negative_loss.sum())

    positive_log_sigmoid_grad =\
        -grad_log_sigmoid(-positive_distance)

    positive_distance_grad =\
        grad_squared_distance(example_embedding, context_embedding)

    positive_grad =\
        gs.repeat(positive_log_sigmoid_grad, dim, axis=-1)\
        * positive_distance_grad

    negative_distance_grad =\
        grad_squared_distance(reshaped_example_embedding, negative_embedding)

    negative_log_sigmoid_grad =\
        grad_log_sigmoid(negative_distance)

    negative_grad = gs.repeat(negative_log_sigmoid_grad, dim, axis=-1)\
        * negative_distance_grad

    example_grad = -(positive_grad + negative_grad.sum(axis=0))

    return total_loss, example_grad


def main():
    """Learning Poincaré graph embedding.

    Learns Poincaré Ball embedding by using Riemannian
    gradient descent algorithm.
    """
    dim = 2
    max_epochs = 100
    lr = .05
    n_negative = 2
    context_size = 1
    karate_graph = gdp.Graph(
        graph_matrix_path='examples/data/graph_karate/karate.txt',
        labels_path='examples/data/graph_karate/karate_labels.txt')

    nb_vertices_by_edges =\
        [len(e_2) for _, e_2 in karate_graph.edges.items()]
    logging.info('Number of edges: %s' % len(karate_graph.edges))
    logging.info('Mean vertices by edges: %s' % (sum(nb_vertices_by_edges, 0) /
                 len(karate_graph.edges)))

    negative_table_parameter = 5
    negative_sampling_table = []

    for i, nb_v in enumerate(nb_vertices_by_edges):
        negative_sampling_table +=\
            ([i] * int((nb_v**(3. / 4.))) * negative_table_parameter)

    negative_sampling_table = gs.array(negative_sampling_table)
    random_walks = karate_graph.random_walk()
    embeddings = gs.random.randn(karate_graph.n_nodes, dim)
    embeddings = embeddings * 0.2

    hyperbolic_manifold = Hyperbolic(2, coords_type='ball')

    colors = {1: 'b', 2: 'r'}
    for epoch in range(max_epochs):
        total_loss = []
        for path in random_walks:

            for example_index in range(len(path)):
                context_index = path[max(0, example_index - context_size):
                                     min(example_index + context_size,
                                     len(path))]
                negative_index =\
                    gs.random.randint(negative_sampling_table.shape[0],
                                      size=(len(context_index),
                                      n_negative))
                negative_index = negative_sampling_table[negative_index]

                example_embedding = embeddings[path[example_index]]

                for k in range(len(negative_index)):
                    context_embedding = embeddings[context_index[k]]
                    negative_embedding = embeddings[negative_index[k]]
                    l, g_ex =\
                        loss(example_embedding, context_embedding,
                             negative_embedding, hyperbolic_manifold)
                    total_loss.append(l)

                    example_to_update = embeddings[path[example_index]]
                    embeddings[path[example_index]] =\
                        hyperbolic_manifold.metric.exp(-lr * g_ex,
                                                       example_to_update)

        logging.info('iteration %d loss_value %f' % (epoch,
                     sum(total_loss, 0) / len(total_loss)))

    circle = visualization.PoincareDisk(point_type='ball')
    plt.figure()
    ax = plt.subplot(111)
    circle.add_points(gs.array([[0, 0]]))
    circle.set_ax(ax)
    circle.draw(ax=ax)
    for i in range(len(embeddings)):
        plt.scatter(embeddings[i][0], embeddings[i][1],
                    c=colors[karate_graph.labels[i][0]])
    plt.show()


if __name__ == "__main__":
    main()
