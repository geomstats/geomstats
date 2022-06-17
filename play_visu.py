"""Script to instantiate visualization_draft."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.visualization_draft import Visualizer2D

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
