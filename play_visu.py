"""Script to instantiate visualization_draft."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.binomial import BinomialDistributions
from geomstats.visualization.information_geometry.visualization_draft_infogeo import (
    Visualizer1D,
    Visualizer2D,
)

"""2D"""

space = BetaDistributions()

visu = Visualizer2D(space)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

visu.plot_grid(
    axs[0, 0], lower_left=gs.array([1.0, 1.0]), upper_right=gs.array([10.0, 10.0])
)
axs[0, 0].title.set_text("geodesic grid of beta distributions")

visu.plot_geodesic_ball(axs[0, 1], center=gs.array([1.0, 1.0]))
axs[0, 1].title.set_text("geodesic ball of beta distributions")

point = space.random_point(5)
vec = space.metric.random_unit_tangent_vec(point)
visu.plot_vector_field(axs[1, 0], point, vec)
axs[1, 0].title.set_text("random vector field in the Beta manifold")
axs[1, 0].axis("equal")

visu.plot_geodesic(axs[1, 1], initial_point=point[0], initial_tangent_vec=vec[0])
axs[1, 1].title.set_text("random geodesic in the Beta manifold")
axs[1, 1].axis("equal")

plt.show()

overlay_fig, overlay_ax = plt.subplots()
visu.overlay_scatter(overlay_ax, point, support=[0, 1])

overlay_ax.title.set_text("overlay scatter of random points on the Beta manifold")
overlay_ax.legend(loc="upper right")

plt.show(block=False)
plt.pause(0.001)  # Pause for interval seconds.
input("hit [enter] to close 2D visualizer.")
plt.close("all")  # all open plots are correctly closed after each run

"""1D"""

n = 10

space = BinomialDistributions(n)

exp = space.metric.exp(gs.array([1.0]), gs.array([0.5]))

fig, axs = plt.subplots(2, figsize=(12, 8))

visu = Visualizer1D(space)

iso = visu.iso_visualizer1D(axs[0], 0, 1, 1000)
iso.isometric_plot_geodesic_ball(0.5, 0.4, label="geodesic ball in isometric immersion")
iso.isometric_scatter(
    [0.4, 0.5, 0.6], s=20, label=f"points of the {n}-binomial manifold"
)
axs[0].set_yticks([])
axs[0].legend(loc="upper right")
axs[0].set_xlabel("p")

center = 0.3
visu.scatter(axs[1], [center], s=50, color="r", label="center of the ball")
visu.plot_geodesic_ball(
    axs[1],
    center,
    0.5,
    linestyle="-",
    label=f"geodesic ball of the {n}-binomial manifold",
)
visu.scatter(axs[1], [0.2, 0.8], s=20, label=f"points of the {n}-binomial manifold")
axs[1].legend(loc="upper right")
axs[1].set_xlabel("p")

plt.show()
