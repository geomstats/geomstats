r"""Compute a geodesic in BHV space between two fully resolved trees with 5 leaves.

In [BHV01], Figure 18, there are depicted the trees T and T', as well as the respective
splits named e1, e2, f1, f1 at the axes. Furthermore, the geodesic between T and T' is
depicted, and in this example, this scenario is reconstructed and the geodesic is
computed.

The geodesic passes a third orthant which one can observe in the printed results of
trees on the geodesic.

Lead author: Jonas Lueg

References
----------
.. [BHV01] Billera, L. J., S. P. Holmes, K. Vogtmann.
    "Geometry of the Space of Phylogenetic Trees."
    Advances in Applied Mathematics,
    volume 27, issue 4, pages 733-767, 2001.
    https://doi.org/10.1006%2Faama.2001.0759
"""

import geomstats.backend as gs
from geomstats.geometry.stratified.bhv_space import Split, Tree, TreeSpace


def main():
    r"""Compute a geodesic in BHV space between two fully resolved trees with 5 leaves.

    Reconstruction of Figure 18 scenario in [BHV01].
    """
    e1 = Split((2, 3), (0, 1, 4))
    e2 = Split((0, 4), (1, 2, 3))
    f1 = Split((0, 1), (2, 3, 4))
    f2 = Split((3, 4), (0, 1, 2))
    split_dict = {e1: "e1", e2: "e2", f1: "f1", f2: "f2"}

    initial_point = Tree(splits=[e1, e2], lengths=[5, 2])
    end_point = Tree(splits=[f1, f2], lengths=[5, 2])

    space = TreeSpace(n_labels=5)

    geod_func = space.metric.geodesic(initial_point=initial_point, end_point=end_point)

    time = gs.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    geod_points = geod_func(time)
    for t, point in zip(time, geod_points):
        p_info = tuple(
            (split_dict[split], round(length, 4))
            for split, length in zip(point.topology.splits, point.lengths)
        )
        print(
            f"Point at time {t}: {p_info[0]}, {p_info[1]}, belongs to BHV space? "
            f"{space.belongs(point, atol=10e-8)}."
        )


if __name__ == "__main__":
    main()
