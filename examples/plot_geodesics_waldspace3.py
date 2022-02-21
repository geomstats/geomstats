"""
Plot three geodesics in Wald space with three labels.

The visualization is done via plotting the three off-diagonal entries of the embedded
points of Wald space that are strictly positive definite symmetric matrices.

The wireframes are two-dimensional boundaries of the embedded Wald space.
The black lines in the origin are the one-dimensional boundaries of Wald space.

Note that this is not an isometric embedding, e.g. the non-marked ends of the wireframes
are infinitely far away from points in the interior.

Lead author: Jonas Lueg
"""

import geomstats.backend as gs
import geomstats.visualization as visualization

from geomstats.geometry.waldspace import WaldSpaceMetric
from geomstats.geometry.trees import Split, Structure, Wald

# Wald space with three labels.
WS3 = WaldSpaceMetric(n=3)


def main():
    """Plot an approximated geodesic from p to q in Wald space. """
    # Construct the tree with three leaves and three edges with one interior vertex.
    sp0 = Split(n=3, part1=(0,), part2=(1, 2))
    sp1 = Split(n=3, part1=(1,), part2=(0, 2))
    sp2 = Split(n=3, part1=(2,), part2=(0, 1))
    st = Structure(n=3, partition=((0, 1, 2),), split_sets=((sp0, sp1, sp2),))

    # Construct the three points in Wald space.
    p1 = Wald(n=3, st=st, x=gs.array([0.1, 0.9, 0.07])).corr
    p2 = Wald(n=3, st=st, x=gs.array([0.08, 0.1, 0.9])).corr
    p3 = Wald(n=3, st=st, x=gs.array([0.3, 0.001, 0.01])).corr

    # Compute the approximations of geodesics.
    proj_args = {'method': 'local', 'btol': 10**-8, 'gtol': 10**-5}
    curve12 = WS3.geodesic(p=p1, q=p2, n_points=20, **proj_args)
    curve23 = WS3.geodesic(p=p2, q=p3, n_points=20, **proj_args)
    curve13 = WS3.geodesic(p=p1, q=p3, n_points=20, **proj_args)
    print(curve12)

    # Plot those paths and points in Wald space embedded into SPD(3).
    ws3_plot = visualization.WaldSpace3()
    for curve in [curve12, curve23, curve13]:
        _curve = WS3.space.to_forest(curve)
        ws3_plot.pass_curve(curve=_curve, label=f"length = {WS3.length(curve)}")
    ws3_plot.pass_points(points=[p1, p2, p3], marker='.',
                         text=[r'$p_1$', r'$p_2$', r'$p_3$'], color='black')
    ws3_plot.show()


if __name__ == "__main__":
    main()
