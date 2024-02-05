"""Learning embedding of graph using Poincare Ball Model."""

import logging

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.datasets.utils import load_karate_graph
from geomstats.geometry.poincare_ball import PoincareBall


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


def grad_squared_distance(point_a, point_b):
    """Gradient of squared hyperbolic distance.

    Gradient of the squared distance based on the
    Ball representation according to point_a

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


def loss(example_embedding, context_embedding, negative_embedding, manifold):
    """Compute loss and grad.

    Compute loss and grad given embedding of the current example,
    embedding of the context and negative sampling embedding.
    """
    n_edges, dim = negative_embedding.shape[0], example_embedding.shape[-1]
    example_embedding = gs.expand_dims(example_embedding, 0)
    context_embedding = gs.expand_dims(context_embedding, 0)

    positive_distance = manifold.metric.squared_dist(
        example_embedding, context_embedding
    )
    positive_loss = log_sigmoid(-positive_distance)

    reshaped_example_embedding = gs.repeat(example_embedding, n_edges, axis=0)

    negative_distance = manifold.metric.squared_dist(
        reshaped_example_embedding, negative_embedding
    )
    negative_loss = log_sigmoid(negative_distance)

    total_loss = -(positive_loss + negative_loss.sum())

    positive_log_sigmoid_grad = -grad_log_sigmoid(-positive_distance)

    positive_distance_grad = grad_squared_distance(example_embedding, context_embedding)

    positive_grad = (
        gs.repeat(positive_log_sigmoid_grad, dim, axis=-1) * positive_distance_grad
    )

    negative_distance_grad = grad_squared_distance(
        reshaped_example_embedding, negative_embedding
    )

    negative_distance = gs.to_ndarray(negative_distance, to_ndim=2, axis=-1)
    negative_log_sigmoid_grad = grad_log_sigmoid(negative_distance)

    negative_grad = negative_log_sigmoid_grad * negative_distance_grad

    example_grad = -(positive_grad + negative_grad.sum(axis=0))

    return total_loss, example_grad


def main():
    """Learning Poincaré graph embedding.

    Learns Poincaré Ball embedding by using Riemannian
    gradient descent algorithm.
    """
    gs.random.seed(1234)
    dim = 2
    max_epochs = 100
    lr = 0.05
    n_negative = 2
    context_size = 1
    karate_graph = load_karate_graph()

    nb_vertices_by_edges = [len(e_2) for _, e_2 in karate_graph.edges.items()]
    logging.info("Number of edges: %s", len(karate_graph.edges))
    logging.info(
        "Mean vertices by edges: %s",
        (sum(nb_vertices_by_edges, 0) / len(karate_graph.edges)),
    )

    negative_table_parameter = 5
    negative_sampling_table = []

    for i, nb_v in enumerate(nb_vertices_by_edges):
        negative_sampling_table += (
            [i] * int((nb_v ** (3.0 / 4.0))) * negative_table_parameter
        )

    negative_sampling_table = gs.array(negative_sampling_table)
    random_walks = karate_graph.random_walk()
    embeddings = gs.random.normal(size=(karate_graph.n_nodes, dim))
    embeddings = embeddings * 0.2

    hyperbolic_manifold = PoincareBall(2)

    colors = {1: "b", 2: "r"}
    for epoch in range(max_epochs):
        total_loss = []
        for path in random_walks:
            for example_index, one_path in enumerate(path):
                context_index = path[
                    max(0, example_index - context_size) : min(
                        example_index + context_size, len(path)
                    )
                ]
                negative_index = gs.random.randint(
                    negative_sampling_table.shape[0],
                    size=(len(context_index), n_negative),
                )
                negative_index = negative_sampling_table[negative_index]

                example_embedding = embeddings[one_path]

                for one_context_i, one_negative_i in zip(context_index, negative_index):
                    context_embedding = embeddings[one_context_i]
                    negative_embedding = embeddings[one_negative_i]
                    l, g_ex = loss(
                        example_embedding,
                        context_embedding,
                        negative_embedding,
                        hyperbolic_manifold,
                    )
                    total_loss.append(l)

                    example_to_update = embeddings[one_path]
                    embeddings[one_path] = hyperbolic_manifold.metric.exp(
                        -lr * g_ex, example_to_update
                    )

        logging.info(
            "iteration %d loss_value %f", epoch, sum(total_loss, 0) / len(total_loss)
        )

    circle = visualization.PoincareDisk(coords_type="ball")
    plt.figure()
    ax = plt.subplot(111)
    circle.add_points(gs.array([[0, 0]]))
    circle.set_ax(ax)
    circle.draw(ax=ax)
    for i_embedding, embedding in enumerate(embeddings):
        plt.scatter(
            embedding[0], embedding[1], c=colors[karate_graph.labels[i_embedding][0]]
        )
    plt.show()


if __name__ == "__main__":
    main()
