"""Run K-means on manifolds for K=2 and Plot the results.

Two random clusters are generated in separate regions of the
manifold. Then K-means is applied using the metric of the manifold.
The points are represented with two distinct colors. For the moment
the example works on the Poincaré Ball and the Hypersphere.
Computed means are marked as green stars.
"""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.kmeans import RiemannianKMeans


def kmean_poincare_ball():
    """Run K-means on the Poincare ball."""
    n_samples = 20
    dim = 2
    n_clusters = 2
    manifold = PoincareBall(dim=dim)

    cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(n_samples, dim))
    cluster_2 = gs.random.uniform(low=0, high=-0.2, size=(n_samples, dim))
    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmeans = RiemannianKMeans(manifold, n_clusters=n_clusters, init="random")

    kmeans.fit(X=data)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

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
        cluster_centers,
        ax=ax,
        space="H2_poincare_disk",
        marker="*",
        color="green",
        s=100,
        coords_type=manifold.default_coords_type,
    )

    ax.set_title("Kmeans on Poincaré Ball Manifold")

    return plt


def kmean_hypersphere():
    """Run K-means on the sphere."""
    n_samples = 50
    dim = 2
    n_clusters = 2
    manifold = Hypersphere(dim)

    # Generate data on north pole
    cluster_1 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)

    # Generate data on south pole
    cluster_2 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)
    cluster_2 = -cluster_2

    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmeans = RiemannianKMeans(manifold, n_clusters, tol=1e-3)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    plt.figure(2)
    colors = ["red", "blue"]

    ax = visualization.plot(data, space="S2", marker=".", color="black")

    for i in range(n_clusters):
        if len(data[labels == i]) > 0:
            ax = visualization.plot(
                points=data[labels == i], ax=ax, space="S2", marker=".", color=colors[i]
            )

    ax = visualization.plot(
        cluster_centers, ax=ax, space="S2", marker="*", s=200, color="green"
    )

    ax.set_title("Kmeans on the sphere")

    return plt


def main():
    """Run K-means on the Poincare ball and the sphere."""
    kmean_poincare_ball()

    plots = kmean_hypersphere()

    plots.show()


if __name__ == "__main__":
    if os.environ.get("GEOMSTATS_BACKEND", "numpy") != "numpy":
        logging.info(
            "Examples with visualizations are only implemented "
            "with numpy backend.\n"
            "To change backend, write: "
            "export GEOMSTATS_BACKEND = 'numpy'."
        )
    else:
        main()
