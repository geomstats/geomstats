"""Based on Le Brigant et al. (2021) Fisher-Rao geometry of Dirichlet distributions."""

import matplotlib.pyplot as plt
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import proj3d

import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions

space = BetaDistributions()

fig = plt.figure(figsize=(12, 8))

n_cells = 30
t = gs.linspace(0, 1, 100)

ax3d = plt.axes(projection="3d")


def to_itg(x):
    """Define function to integrate to define eta."""
    return gs.sqrt(gs.polygamma(1, x))


def eta(x):
    """Define isometric embedding.

    References
    ----------
    [BPP2021] Le Brigant, A., & Preston, C. & Puechmorel, S. (2021).
        Fisher-Rao geometry of Dirichlet distributions (p. 5).
    """
    return integrate.quad(to_itg, 1, x)[0]


ALPHA = gs.linspace(0.1, 3.1, n_cells)
BETA = gs.linspace(0.1, 3.1, n_cells)

eta1 = [eta(alpha) for alpha in ALPHA]
eta2 = [eta(beta) for beta in BETA]
eta3 = [eta(alpha + beta) for alpha in ALPHA for beta in BETA]

ETA1, ETA2 = gs.meshgrid(eta1, eta2)
ETA3 = gs.reshape(eta3, (n_cells, n_cells))

base_point1 = gs.array([1, 1])
end_point1 = gs.array([2, 3])
geodesic1 = space.metric.geodesic(initial_point=base_point1, end_point=end_point1)
geodesic1 = geodesic1(t)
iso_geodesic1 = gs.array(
    [
        [eta(geodesic1[i, 0]), eta(geodesic1[i, 1]), eta(gs.sum(geodesic1[i]))]
        for i in range(100)
    ]
)
ax3d.plot(iso_geodesic1[:, 0], iso_geodesic1[:, 1], iso_geodesic1[:, 2])

base_point2 = gs.array([2, 1])
end_point2 = gs.array([1, 2])
geodesic2 = space.metric.geodesic(initial_point=base_point2, end_point=end_point2)
geodesic2 = geodesic2(t)
iso_geodesic2 = gs.array(
    [
        [eta(geodesic2[i, 0]), eta(geodesic2[i, 1]), eta(gs.sum(geodesic2[i]))]
        for i in range(100)
    ]
)
ax3d.plot(iso_geodesic2[:, 0], iso_geodesic2[:, 1], iso_geodesic2[:, 2])

ax3d.plot_surface(ETA1, ETA2, ETA3, cmap=plt.cm.gray, alpha=0.7)
ax3d.set_title("Isometric Visualizer of Beta distributions in Minkowski space")
ax3d.set_xlabel("$\\eta(\\alpha)$")
ax3d.set_ylabel("$\\eta(\\beta)$")
ax3d.set_zlabel("$\\eta(\\alpha+\\beta)$")


def distance(point, event):
    """Return distance between mouse position and given data point."""
    assert point.shape == (3,), (
        "distance: point.shape is wrong: %s, must be (3,)" % point.shape
    )

    # Project 3d data space to 2d data space
    x2, y2, _ = proj3d.proj_transform(
        point[0], point[1], point[2], plt.gca().get_proj()
    )
    # Convert 2d data space to 2d screen space
    x3, y3 = ax3d.transData.transform((x2, y2))

    return gs.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)


def calcClosestDatapoint(X, event):
    """Calculate which data point is closest to the mouse position."""
    distances = [distance(X[i, 0:3], event) for i in range(X.shape[0])]
    return gs.argmin(distances)


def annotatePlot(X, index):
    """Create popover label in 3d chart."""
    # If we have previously displayed another label, remove it first
    if hasattr(annotatePlot, "label"):
        annotatePlot.label.remove()
    # Get data point from array of points X, at position index
    x2, y2, _ = proj3d.proj_transform(
        X[index, 0], X[index, 1], X[index, 2], ax3d.get_proj()
    )
    annotatePlot.label = plt.annotate(
        f"$\\alpha = {str(ALPHA[index%n_cells])[:5]},"
        + f"\\beta = {str(BETA[index//n_cells])[:5]}$",
        xy=(x2, y2),
        xytext=(-20, 20),
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    fig.canvas.draw()


X = gs.transpose(gs.array([gs.flatten(ETA1), gs.flatten(ETA2), gs.flatten(ETA3)]))


def onMouseMotion(event):
    """Show text annotation over data point closest to mouse when mouse is moved."""
    closestIndex = calcClosestDatapoint(X, event)
    annotatePlot(X, closestIndex)


fig.canvas.mpl_connect("motion_notify_event", onMouseMotion)  # on mouse motion
plt.show()
