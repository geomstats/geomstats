"""Plot the pole ladder scheme for parallel transport on S2.

Sample a point on S2 and two tangent vectors to transport one along the
other.

"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere


space = Hypersphere(2)
metric = space.metric

n_steps = 4

point = space.random_uniform(1)
tan_b = space.random_uniform(1)
tan_b = space.projection_to_tangent_space(tan_b, point) * n_steps / 2
tan_a = space.random_uniform(1)
tan_a = space.projection_to_tangent_space(tan_a, point) / 4


pole_ladder, trajectory = metric.pole_ladder_parallel_transport(
    tan_a, tan_b, point, n_steps=n_steps, return_geodesics=True)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sphere_visu = visualization.Sphere(n_meridians=30)
ax = sphere_visu.set_ax(ax=ax)

n_points = 10
t = gs.linspace(0, 1, n_points)
t_main = gs.linspace(0, 1, n_points * 4)
for points in trajectory:
    main_geodesic, diagonal, final_geodesic = points
    sphere_visu.draw_points(ax, main_geodesic(t_main), marker='o', c='b', s=2)
    sphere_visu.draw_points(ax, diagonal(-t), marker='o', c='r', s=2)
    sphere_visu.draw_points(ax, diagonal(t), marker='o', c='r', s=2)
    sphere_visu.draw_points(ax, final_geodesic(-t), marker='o', c='g', s=2)
    sphere_visu.draw_points(ax, final_geodesic(t), marker='o', c='g', s=2)

V = gs.concatenate([tan_b / n_steps, tan_a, pole_ladder])
origin = gs.concatenate([point, point, final_geodesic(gs.array([0]))])

ax.quiver(
    origin[:, 0], origin[:, 1], origin[:, 2],
    V[:, 0], V[:, 1], V[:, 2],
    color=['black', 'black', 'black'],
    linewidth=2)

sphere_visu.draw(ax, linewidth=1)
plt.show()
