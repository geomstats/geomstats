"""Learning embedding of graph using Poincare Ball Model."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.datasets.prepare_graph_data import HyperbolicEmbedding
from geomstats.datasets.utils import load_karate_graph
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.learning.kmedoids import RiemannianKMedoids


def main():
    """Learning Poincaré graph embedding.

    Learns Poincaré Ball embedding by using Riemannian
    gradient descent algorithm. Then K-means is applied
    to learn labels of each data sample.
    """
    gs.random.seed(1234)

    karate_graph = load_karate_graph()
    hyperbolic_embedding = HyperbolicEmbedding(max_epochs=3)
    embeddings = hyperbolic_embedding.embed(karate_graph)

    colors = {1: 'b', 2: 'r'}
    group_1 = mpatches.Patch(color=colors[1], label='Group 1')
    group_2 = mpatches.Patch(color=colors[2], label='Group 2')

    circle = visualization.PoincareDisk(point_type='ball')

    _, ax = plt.subplots(figsize=(8, 8))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    circle.set_ax(ax)
    circle.draw(ax=ax)
    for i_embedding, embedding in enumerate(embeddings):
        x_coords = embedding[0]
        y_coords = embedding[1]
        pt_id = i_embedding
        plt.scatter(
            x_coords, y_coords, c=colors[karate_graph.labels[pt_id][0]], s=150)
        ax.annotate(pt_id, (x_coords, y_coords))

    plt.tick_params(
        which='both')
    plt.title('Poincare Ball Embedding of the Karate Club Network')
    plt.legend(handles=[group_1, group_2])
    plt.show()

    n_clusters = 2

    kmeans = RiemannianKMeans(
        metric=hyperbolic_embedding.manifold.metric,
        n_clusters=n_clusters,
        init='random')

    centroids = kmeans.fit(X=embeddings)
    labels = kmeans.predict(X=embeddings)

    colors = ['g', 'c', 'm']
    circle = visualization.PoincareDisk(point_type='ball')
    _, ax2 = plt.subplots(figsize=(8, 8))
    circle.set_ax(ax2)
    circle.draw(ax=ax2)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    group_1_predicted = mpatches.Patch(
        color=colors[0], label='Predicted Group 1')
    group_2_predicted = mpatches.Patch(
        color=colors[1], label='Predicted Group 2')
    group_centroids = mpatches.Patch(
        color=colors[2], label='Cluster centroids')

    for _ in range(n_clusters):
        for i_embedding, embedding in enumerate(embeddings):
            x_coords = embedding[0]
            y_coords = embedding[1]
            pt_id = i_embedding
            if labels[i_embedding] == 0:
                color = colors[0]
            else:
                color = colors[1]
            plt.scatter(
                x_coords, y_coords,
                c=color,
                s=150
            )
            ax2.annotate(pt_id, (x_coords, y_coords))

    for _, centroid in enumerate(centroids):
        x_coords = centroid[0]
        y_coords = centroid[1]
        plt.scatter(
            x_coords, y_coords,
            c=colors[2],
            marker='*',
            s=150,
        )

    plt.title('K-means applied to Karate club embedding')
    plt.legend(handles=[group_1_predicted, group_2_predicted, group_centroids])
    plt.show()

    kmedoid = RiemannianKMedoids(
        metric=hyperbolic_embedding.manifold.metric,
        n_clusters=n_clusters,
        init='random', n_jobs=2)

    centroids = kmedoid.fit(data=embeddings, max_iter=100)
    labels = kmedoid.predict(data=embeddings)

    colors = ['g', 'c', 'm']
    circle = visualization.PoincareDisk(point_type='ball')
    _, ax2 = plt.subplots(figsize=(8, 8))
    circle.set_ax(ax2)
    circle.draw(ax=ax2)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    group_1_predicted = mpatches.Patch(
        color=colors[0], label='Predicted Group 1')
    group_2_predicted = mpatches.Patch(
        color=colors[1], label='Predicted Group 2')
    group_centroids = mpatches.Patch(
        color=colors[2], label='Cluster centroids')

    for _ in range(n_clusters):
        for i_embedding, embedding in enumerate(embeddings):
            x_coords = embedding[0]
            y_coords = embedding[1]
            pt_id = i_embedding
            if labels[i_embedding] == 0:
                color = colors[0]
            else:
                color = colors[1]
            plt.scatter(
                x_coords, y_coords,
                c=color,
                s=150
            )
            ax2.annotate(pt_id, (x_coords, y_coords))

    for _, centroid in enumerate(centroids):
        x_coords = centroid[0]
        y_coords = centroid[1]
        plt.scatter(
            x_coords, y_coords,
            c=colors[2],
            marker='*',
            s=150,
        )

    plt.title('K-Medoids applied to Karate club embedding')
    plt.legend(handles=[group_1_predicted, group_2_predicted, group_centroids])
    plt.show()


if __name__ == '__main__':
    main()
