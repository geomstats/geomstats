"""Run K-medoids on manifolds for K=2 and Plot the results.

Two random clusters are generated in separate regions of the
manifold. Then K-medoids is applied using the metric of the manifold.
The points are represented with two distinct colors. For the moment
the example works on the Poincaré Ball and the Hypersphere.
Computed centroids are marked as green stars.
"""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.kmedoids import RiemannianKMedoids


def kmedoids_poincare_ball():
    """Run K-medoids on the Poincare ball."""
    n_samples = 20
    dim = 2
    n_clusters = 2
    manifold = PoincareBall(dim=dim)

    cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(n_samples, dim))
    cluster_2 = gs.random.uniform(low=-0.2, high=0, size=(n_samples, dim))
    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmedoids = RiemannianKMedoids(
        manifold, n_clusters=n_clusters, max_iter=100, init="random"
    )

    kmedoids.fit(X=data)
    centroids = kmedoids.centroids_
    labels = kmedoids.labels_

    plt.figure(1)
    colors = ["red", "blue"]

    ax = visualization.plot(
        data,
        space="H2_poincare_disk",
        marker=".",
        color="black",
        coords_type=manifold.default_coords_type,
    )

    for i in range(n_clusters):
        ax = visualization.plot(
            data[labels == i],
            ax=ax,
            space="H2_poincare_disk",
            marker=".",
            color=colors[i],
            coords_type=manifold.default_coords_type,
        )

    ax = visualization.plot(
        centroids,
        ax=ax,
        space="H2_poincare_disk",
        marker="*",
        color="green",
        s=100,
        coords_type=manifold.default_coords_type,
    )

    ax.set_title("Kmedoids on Poincaré Ball Manifold")

    return plt


def kmedoids_hypersphere():
    """Run K-medoids on the sphere."""
    n_samples = 50
    dim = 2
    n_clusters = 2
    manifold = Hypersphere(dim)

    # Generate data on north pole
    cluster_1 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)

    # Generate data on south pole
    cluster_2 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)
    for point in cluster_2:
        point[2] = -point[2]

    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmedoids = RiemannianKMedoids(manifold, n_clusters=n_clusters)
    kmedoids.fit(X=data)
    centroids = kmedoids.centroids_
    labels = kmedoids.labels_

    plt.figure(2)
    colors = ["red", "blue"]

    ax = visualization.plot(data, space="S2", marker=".", color="black")

    for i in range(n_clusters):
        if len(data[labels == i]) > 0:
            ax = visualization.plot(
                points=data[labels == i], ax=ax, space="S2", marker=".", color=colors[i]
            )

    ax = visualization.plot(
        centroids, ax=ax, space="S2", marker="*", s=200, color="green"
    )

    ax.set_title("Kmedoids on Hypersphere Manifold")

    return plt


def main():
    """Run K-medoids on the Poincare ball and the sphere."""
    kmedoids_poincare_ball()

    plots = kmedoids_hypersphere()

    plots.show()


if __name__ == "__main__":
    compatible_backends = ["numpy", "pytorch"]

    if os.environ.get("GEOMSTATS_BACKEND", "numpy") not in compatible_backends:
        logging.info(
            "K-Medoids example is implemented"
            "with numpy or pytorch backend.\n"
            "To change backend, write: "
            "export GEOMSTATS_BACKEND = 'numpy'."
        )
    else:
        main()
