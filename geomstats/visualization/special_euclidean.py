"""Visualization for Geometric Statistics."""
import logging

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SE3_GROUP = SpecialEuclidean(n=3, point_type="vector")
SE2_GROUP = SpecialEuclidean(n=2, point_type="matrix")
SE2_VECT = SpecialEuclidean(n=2, point_type="vector")
SO3_GROUP = SpecialOrthogonal(n=3, point_type="vector")
S1 = Hypersphere(dim=1)
S2 = Hypersphere(dim=2)
H2 = Hyperboloid(dim=2)
POINCARE_HALF_PLANE = PoincareHalfSpace(dim=2)
M32 = Matrices(m=3, n=2)
S32 = PreShapeSpace(k_landmarks=3, m_ambient=2)
METRIC_S32 = KendallShapeMetric(k_landmarks=3, m_ambient=2)
M33 = Matrices(m=3, n=3)
S33 = PreShapeSpace(k_landmarks=3, m_ambient=3)
METRIC_S33 = KendallShapeMetric(k_landmarks=3, m_ambient=3)

AX_SCALE = 1.2


class SpecialEuclidean2:
    """Class used to plot points in the 2d special euclidean group."""

    def __init__(self, points=None, point_type="matrix"):
        self.points = []
        self.point_type = point_type
        if points is not None:
            self.add_points(points)

    @staticmethod
    def set_ax(ax=None, x_lim=None, y_lim=None):
        if ax is None:
            ax = plt.subplot()
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        return ax

    def add_points(self, points):
        if self.point_type == "vector":
            points = SE2_VECT.matrix_from_vector(points)
        if not gs.all(SE2_GROUP.belongs(points)):
            logging.warning("Some points do not belong to SE2.")
        if not isinstance(points, list):
            points = list(points)
        self.points.extend(points)

    def draw_points(self, ax, points=None, **kwargs):
        if points is None:
            points = gs.array(self.points)
        translation = points[..., :2, 2]
        frame_1 = points[:, :2, 0]
        frame_2 = points[:, :2, 1]
        ax.quiver(
            translation[:, 0],
            translation[:, 1],
            frame_1[:, 0],
            frame_1[:, 1],
            width=0.005,
            color="b",
        )
        ax.quiver(
            translation[:, 0],
            translation[:, 1],
            frame_2[:, 0],
            frame_2[:, 1],
            width=0.005,
            color="r",
        )
        ax.scatter(translation[:, 0], translation[:, 1], s=16, **kwargs)
