import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.waldspace import WaldSpace
from geomstats.geometry.trees import Split, Structure, Wald
WS3 = WaldSpace(n=3)


def main():
    """Plot an approximated geodesic from p to q in Wald space. """
    sp0 = Split(n=3, part1=(0,), part2=(1, 2))
    sp1 = Split(n=3, part1=(1,), part2=(0, 2))
    sp2 = Split(n=3, part1=(2,), part2=(0, 1))
    st = Structure(n=3, partition=((0, 1, 2),), split_sets=((sp0, sp1, sp2),))

    p = Wald(n=3, st=st, x=gs.array([0.1, 0.9, 0.07]))
    q = Wald(n=3, st=st, x=gs.array([0.08, 0.1, 0.9]))

    proj_args = {'method': 'local', 'btol': 10**-8, 'gtol': 10**-5}
    curve = WS3.geodesic(p=p, q=q, n_points=20, **proj_args)

    ws3_plot = visualization.WaldSpace3()
    ws3_plot.pass_curve(curve=curve, label="geodesic between p and q")
    ws3_plot.show()


if __name__ == "__main__":
    main()
