"""Class for the tripod or three-spider.

Lead authors: Anna Calissano & Jonas Lueg
"""

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.stratified_geometry.metric_spaces import LengthSpace, Point


class TriPoint(Point):
    """Class for points of the space Tripod.

    Parameters
    ----------
    s : int
        The stratum, an integer indicating the stratum the point lies in. If zero, then
        the point is on the origin.
    x : float
        A positive number, the coordinate of the point. It must be zero if and only if
        the stratum is zero, i.e. the origin.
    """

    def __init__(self, s, x):
        super(Point, self).__init__()
        if s not in [0, 1, 2, 3]:
            raise ValueError("Number of the stratum must be in [0, 1, 2, 3].")
        if s == 0 and x != 0:
            raise ValueError("If the stratum is zero, x must be zero.")
        if s != 0 and x <= 0:
            raise ValueError("If the stratum is 1, 2 or 3, then x must be positive.")
        self.s = s
        self.x = x

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"s{self.s}: {np.round(self.x, 5)}"

    def __hash__(self):
        """Return the hash of the instance."""
        return hash((self.s, self.x))


class TriPod(LengthSpace[TriPoint]):
    """Tripod as a length space induced from Euclidean distance."""

    def __init__(self):
        super(TriPod, self).__init__()
        self.e = Euclidean(dim=1).metric

    def random_point(self) -> TriPoint:
        """Compute a random point of the tripod space."""
        _s = gs.random.randint(low=0, high=4)
        if _s == 0:
            return TriPoint(s=_s, x=0.0)
        _x = gs.abs(gs.random.normal(0, 1))
        return TriPoint(s=_s, x=_x)

    def dist(self, p: TriPoint, q: TriPoint, **kwargs):
        """Compute the distance between two points."""
        if p.s == q.s or p.s == 0 or q.s == 0:
            return gs.abs(p.x - q.x)
        return p.x + q.x

    def geodesic(self, p: TriPoint, q: TriPoint, **kwargs):
        """Compute points on the geodesic between two points.

        t : float between 0 and 1
            The portion of time of the returned point.
        """
        t = kwargs["t"] if "t" in kwargs else 0.5
        if p.s == q.s or p.s == 0 or q.s == 0:
            s = p.s if q.s == 0 else q.s
            g = self.e.geodesic(
                initial_point=gs.array([p.x]), end_point=gs.array([q.x])
            )
            return TriPoint(s=s, x=float(g(t)))
        g = self.e.geodesic(initial_point=gs.array([-p.x]), end_point=gs.array([q.x]))
        x = float(g(t))
        if x < 0:
            return TriPoint(s=p.s, x=-x)
        if x > 0:
            return TriPoint(s=q.s, x=x)
        return TriPoint(s=0, x=0.0)


if __name__ == "__main__":
    t3 = TriPod()
    for i in range(10):
        p = t3.random_point()
        q = t3.random_point()
        dist = t3.dist(p, q)
        r = t3.geodesic(p, q, t=0.5)
        print(f"Distance between {p} and {q} is {np.round(dist, 5)}, midpoint is {r}.")
