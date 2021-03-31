"""Plot the pole ladder scheme for parallel transport on S2.

Sample a point on S2 and two tangent vectors to transport one along the
other.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


SPACE = Hypersphere(2)
METRIC = SPACE.metric
ROTATIONS = SpecialOrthogonal(3, 'vector')

N_STEPS = 4
N_POINTS = 10

gs.random.seed(1)


def main():
    """Compute pole ladder and plot the construction."""
    base_point = SPACE.random_uniform(1)
    tangent_vec_b = SPACE.random_uniform(1)
    tangent_vec_b = SPACE.to_tangent(tangent_vec_b, base_point)
    tangent_vec_b = tangent_vec_b / gs.linalg.norm(tangent_vec_b)

    rotation_vector = gs.pi / 2 * base_point
    rotation_matrix = ROTATIONS.matrix_from_rotation_vector(rotation_vector)
    tangent_vec_a = gs.dot(rotation_matrix, tangent_vec_b)
    tangent_vec_b *= 3. / 2.

    ladder = METRIC.ladder_parallel_transport(
        tangent_vec_a,
        tangent_vec_b,
        base_point,
        n_rungs=N_STEPS,
        return_geodesics=True)

    pole_ladder = ladder['transported_tangent_vec']
    trajectory = ladder['trajectory']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    sphere_visu = visualization.Sphere(n_meridians=30)
    ax = sphere_visu.set_ax(ax=ax)

    t = gs.linspace(0., 1., N_POINTS)
    t_main = gs.linspace(0., 1., N_POINTS * 4)
    for points in trajectory:
        main_geodesic, diagonal, final_geodesic = points
        sphere_visu.draw_points(
            ax, main_geodesic(t_main), marker='o', c='b', s=2)
        sphere_visu.draw_points(ax, diagonal(-t), marker='o', c='r', s=2)
        sphere_visu.draw_points(ax, diagonal(t), marker='o', c='r', s=2)

    tangent_vectors = gs.stack(
        [tangent_vec_b, tangent_vec_a, pole_ladder], axis=0) / N_STEPS

    base_point = gs.to_ndarray(base_point, to_ndim=2)
    origin = gs.concatenate(
        [base_point, base_point, final_geodesic(0.)], axis=0)
    ax.quiver(
        origin[:, 0], origin[:, 1], origin[:, 2],
        tangent_vectors[:, 0], tangent_vectors[:, 1], tangent_vectors[:, 2],
        color=['black', 'black', 'black'],
        linewidth=2)

    sphere_visu.draw(ax, linewidth=1)
    plt.show()


if __name__ == '__main__':
    main()
