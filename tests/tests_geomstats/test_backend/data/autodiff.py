import functools

import pytest
from polpo.testing.data import CaseData

import geomstats.backend as gs
from geomstats.test.test_case import pytorch_backend

from ._base import VectorizableCaseData

rand = gs.random.rand


def _square(x):
    """Square a scalar."""
    return x**2


def _first_squared_plus_second(x):
    """Add the second coordinate to the squared first one."""
    return x[..., 0] ** 2 + x[..., 1]


def _mat_entries_polynomial(x):
    """Compute a polynomial of three entries of a matrix."""
    return x[..., 0, 0] ** 2 + x[..., 0, 1] + 2 * x[..., 1, 0]


def _scalar_squared_dist(x, y):
    """Compute the squared distance between scalars."""
    return (x - y) ** 2


def _squared_dist_to_scalar(x, scalar):
    """Compute the squared distance of a vector to a constant vector."""
    return gs.sum((x - scalar[..., None]) ** 2, axis=-1)


def _squared_dist(x, y):
    """Compute the squared distance between vectors."""
    return gs.sum((x - y) ** 2, axis=-1)


def _mat_squared_dist(x, y):
    """Compute the squared distance between matrices."""
    return gs.sum((x - y) ** 2, axis=(-2, -1))


def _dot_with_square(x, y):
    """Compute the dot product of a vector with the square of another."""
    return gs.sum(x * y**2, axis=-1)


def _matvec_sum(mat, vec):
    """Sum the entries of a matrix-vector product."""
    return gs.sum(gs.matvec(mat, vec), axis=-1)


def _expm_sum(x):
    """Sum the entries of a matrix exponential."""
    return gs.sum(gs.linalg.expm(x), axis=(-2, -1))


def _logm_sum(x):
    """Sum the entries of a matrix logarithm."""
    return gs.sum(gs.linalg.logm(x), axis=(-2, -1))


def _sylvester_sum(x):
    """Sum the entries of the solution of ``x X + X x = 1``, of ones."""
    ones = gs.ones(x.shape[-2:])
    return gs.sum(gs.linalg.solve_sylvester(x, x, ones), axis=(-2, -1))


def _clip_sum(x):
    """Sum the entries of a vector, clipped from below with a keyword."""
    return gs.sum(gs.clip(x, min=0.5), axis=-1)


def _weighted_scatter_add_sum(src):
    """Sum, with weights, the values of src added at fixed indices."""
    index = gs.broadcast_to(gs.array([0, 2, 0]), src.shape)
    out = gs.scatter_add(gs.zeros(src.shape), dim=-1, index=index, src=src)
    return gs.sum(gs.array([1.0, 2.0, 3.0]) * out, axis=-1)


def _diagonal_sum(x):
    """Sum the diagonal entries of a matrix."""
    return gs.sum(gs.diagonal(x), axis=-1)


def _weighted_transpose_sum(x):
    """Sum the entries of the transpose of a matrix, weighted."""
    weights = gs.array([[1.0, 2.0], [3.0, 4.0]])
    return gs.sum(weights * gs.matrix_transpose(x), axis=(-2, -1))


def _six_times(x):
    """Return six times the input, a fake gradient."""
    return 6 * x


def _six_times_difference(x, y):
    """Return six times the difference, a fake gradient."""
    return 6 * (x - y)


def _six_times_reversed_difference(x, y):
    """Return six times the reversed difference, a fake gradient."""
    return 6 * (y - x)


@gs.autodiff.custom_gradient(_six_times)
def _sum_of_squares_with_fake_grad(x):
    """Sum the squares of the coordinates, with a fake gradient."""
    return gs.sum(x**2, axis=-1)


@gs.autodiff.custom_gradient(_six_times_difference, _six_times_reversed_difference)
def _squared_dist_with_fake_grads(x, y):
    """Compute the squared distance between vectors, with fake gradients."""
    return gs.sum((x - y) ** 2, axis=-1)


@gs.autodiff.custom_gradient(_six_times_difference, _six_times_reversed_difference)
def _mat_squared_dist_with_fake_grads(x, y):
    """Compute the squared distance between matrices, with fake gradients."""
    return gs.sum((x - y) ** 2, axis=(-2, -1))


@gs.autodiff.custom_gradient(_six_times)
def _squared_sum_with_fake_grad(x):
    """Square the sum of the coordinates, with a fake gradient."""
    return gs.sum(x, axis=-1) ** 2


@gs.autodiff.custom_gradient(_six_times)
def _mat_squared_sum_with_fake_grad(x):
    """Square the sum of the entries, with a fake gradient."""
    return gs.sum(x, axis=(-2, -1)) ** 2


def _cubed_squared_sum_with_fake_grad(x):
    """Cube ``_squared_sum_with_fake_grad``."""
    return _squared_sum_with_fake_grad(x) ** 3


def _mat_cubed_squared_sum_with_fake_grad(x):
    """Cube ``_mat_squared_sum_with_fake_grad``."""
    return _mat_squared_sum_with_fake_grad(x) ** 3


def _sphere_immersion(point):
    """Immerse spherical coordinates in the sphere of radius 4."""
    theta, phi = point[..., 0], point[..., 1]
    x = gs.sin(theta) * gs.cos(phi)
    y = gs.sin(theta) * gs.sin(phi)
    z = gs.cos(theta)
    return 4.0 * gs.stack([x, y, z], axis=-1)


def _sphere_immersion_jacobian(point):
    """Compute the closed-form jacobian of ``_sphere_immersion``."""
    theta, phi = point
    return 4.0 * gs.array(
        [
            [gs.cos(theta) * gs.cos(phi), -gs.sin(theta) * gs.sin(phi)],
            [gs.cos(theta) * gs.sin(phi), gs.sin(theta) * gs.cos(phi)],
            [-gs.sin(theta), 0.0],
        ]
    )


def _first_component(point):
    """Compute the first component of ``_sphere_immersion``."""
    theta, phi = point[..., 0], point[..., 1]
    return 4.0 * gs.sin(theta) * gs.cos(phi)


def _first_component_hessian(point):
    """Compute the closed-form hessian of ``_first_component``."""
    theta, phi = point
    return 4.0 * gs.array(
        [
            [-gs.sin(theta) * gs.cos(phi), -gs.cos(theta) * gs.sin(phi)],
            [-gs.cos(theta) * gs.sin(phi), -gs.sin(theta) * gs.cos(phi)],
        ]
    )


def _first_component_third_derivative(point):
    """Compute the closed-form third derivative of ``_first_component``."""
    theta, phi = point
    d_theta = 4.0 * gs.array(
        [
            [-gs.cos(theta) * gs.cos(phi), gs.sin(theta) * gs.sin(phi)],
            [gs.sin(theta) * gs.sin(phi), -gs.cos(theta) * gs.cos(phi)],
        ]
    )
    d_phi = 4.0 * gs.array(
        [
            [gs.sin(theta) * gs.sin(phi), -gs.cos(theta) * gs.cos(phi)],
            [-gs.cos(theta) * gs.cos(phi), gs.sin(theta) * gs.sin(phi)],
        ]
    )
    return gs.stack([d_theta, d_phi], axis=-1)


def _autodiff_first_component_jacobian(point):
    """Compute the jacobian of ``_first_component`` with autodiff."""
    return gs.autodiff.jacobian(_first_component)(point)


def _autodiff_first_component_hessian(point):
    """Compute the hessian of ``_first_component`` with autodiff."""
    return gs.autodiff.hessian(_first_component)(point)


def _coordinates_product(point):
    """Multiply the two coordinates of a point."""
    return point[0] * point[1]


def _autodiff_coordinates_product_hessian(point):
    """Compute the hessian of ``_coordinates_product`` with autodiff."""
    return gs.autodiff.hessian(_coordinates_product)(point)


def _square_and_product(point):
    """Map a point to its squared first coordinate and coordinates product."""
    return gs.stack([point[..., 0] ** 2, point[..., 0] * point[..., 1]], axis=-1)


def _cube(x):
    """Cube each entry."""
    return x**3


def _half_squared_sum_hessian(n):
    """Compute the hessian of ``sum(x @ x) / 2`` for n x n matrices."""
    eye = gs.eye(n)
    return (eye[None, :, :, None] + eye[:, None, None, :]) / 2


def _mat_sum_of_cubes(x):
    """Sum the cubes of the entries of a matrix."""
    return gs.sum(x**3, axis=(-2, -1))


def _cube_jacobian_sum(x):
    """Sum the entries of the jacobian of ``_cube``, i.e. ``3 * sum(x**2)``."""
    return gs.sum(gs.autodiff.jacobian(_cube)(x), axis=(-2, -1))


def _get_derivative_func_getter(name):
    """Get a getter of the function computing a derivative of a case.

    Parameters
    ----------
    name : str
        Name of the function of ``gs.autodiff`` computing the derivative.

    Returns
    -------
    get_derivative_func : callable
        Function of a case returning the function of its unrepeated inputs
        computing the derivative of its ``func``. The ``point_ndim`` and
        ``func_out_ndim`` of the case are passed if set.
    """

    def get_derivative_func(case):
        kwargs = {
            key: case[key] for key in ("point_ndim", "func_out_ndim") if key in case
        }
        return lambda **inputs: getattr(gs.autodiff, name)(case["func"], **kwargs)(
            *inputs.values()
        )

    return get_derivative_func


def _sqrtm_sum(x):
    """Sum the entries of a matrix square root."""
    return gs.sum(gs.linalg.sqrtm(x), axis=(-2, -1))


def _real_sqrtm_sum(x):
    """Sum the real parts of the entries of a matrix square root."""
    return gs.real(gs.sum(gs.linalg.sqrtm(x), axis=(-2, -1)))


def _fractional_matrix_power_sum(x):
    """Sum the entries of a fractional power of a matrix."""
    return gs.sum(gs.linalg.fractional_matrix_power(x, 0.5), axis=(-2, -1))


class NotDifferentiableTestData(CaseData):
    def value_and_grad_raises_test_data(self):
        x = gs.array([[2.0, 1.0], [1.0, 2.0]])
        return [
            dict(func=_fractional_matrix_power_sum, x=x),
        ]


class NumpyRaisesTestData(CaseData):
    def raises_test_data(self):
        data = [
            dict(autodiff_func=gs.autodiff.value_and_grad),
            dict(autodiff_func=gs.autodiff.value_and_jacobian),
            dict(autodiff_func=gs.autodiff.jacobian),
            dict(autodiff_func=gs.autodiff.hessian),
            dict(autodiff_func=gs.autodiff.jacobian_and_hessian),
            dict(autodiff_func=gs.autodiff.value_jacobian_and_hessian),
        ]
        return data


class _ValueAndGradTestData(VectorizableCaseData):
    def _get_expected_func(self, case):
        # lazy, as value_and_grad raises without autodiff
        return lambda **inputs: gs.autodiff.value_and_grad(
            case["func"],
            argnums=case.get("argnums", 0),
            point_ndims=case.get("point_ndims", 1),
        )(*inputs.values())


class AutodiffTestData(_ValueAndGradTestData):
    @pytest.mark.smoke
    def value_and_grad_test_data(self):
        data = [
            dict(
                func=_square,
                x=gs.array(3.0),
                point_ndims=0,
                expected_value=9.0,
                expected_grad=6.0,
            ),
            dict(
                func=_first_squared_plus_second,
                x=gs.array([3.0, 2.0]),
                expected_value=11.0,
                expected_grad=gs.array([6.0, 1.0]),
            ),
            dict(
                func=_mat_entries_polynomial,
                x=gs.array([[4.0, 3.0], [2.0, 1.0]]),
                point_ndims=2,
                expected_value=23.0,
                expected_grad=gs.array([[8.0, 1.0], [2.0, 0.0]]),
            ),
            dict(
                func=_squared_dist,
                x=gs.array([1.0, 2.0]),
                y=gs.array([2.0, 3.0]),
                argnums=0,
                expected_value=2.0,
                expected_grad=gs.array([-2.0, -2.0]),
            ),
            dict(
                func=_squared_dist,
                x=gs.array([1.0, 2.0]),
                y=gs.array([2.0, 3.0]),
                argnums=(0, 1),
                expected_value=2.0,
                expected_grad=(
                    gs.array([-2.0, -2.0]),
                    gs.array([2.0, 2.0]),
                ),
            ),
            dict(
                func=_mat_squared_dist,
                x=gs.array([[1.0, 3.0], [2.0, 3.0]]),
                y=gs.array([[2.0, 5.0], [0.0, 4.0]]),
                argnums=1,
                point_ndims=2,
                expected_value=10.0,
                expected_grad=gs.array([[2.0, 4.0], [-4.0, 2.0]]),
            ),
            dict(
                func=_mat_squared_dist,
                x=gs.array([[1.0, 3.0], [2.0, 3.0]]),
                y=gs.array([[2.0, 5.0], [0.0, 4.0]]),
                argnums=(0, 1),
                point_ndims=2,
                expected_value=10.0,
                expected_grad=(
                    gs.array([[-2.0, -4.0], [4.0, -2.0]]),
                    gs.array([[2.0, 4.0], [-4.0, 2.0]]),
                ),
            ),
            dict(
                func=_matvec_sum,
                x=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                y=gs.array([5.0, 6.0]),
                point_ndims=(2, 1),
                argnums=(0, 1),
                expected_value=56.0,
                expected_grad=(
                    gs.array([[5.0, 6.0], [5.0, 6.0]]),
                    gs.array([4.0, 6.0]),
                ),
            ),
            dict(
                func=_matvec_sum,
                x=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                y=gs.array([5.0, 6.0]),
                point_ndims=(2, 1),
                argnums=0,
                expected_value=56.0,
                expected_grad=gs.array([[5.0, 6.0], [5.0, 6.0]]),
            ),
            dict(
                func=gs.vecdot,
                x=gs.array([1.0, 2.0]),
                y=gs.array([3.0, 4.0]),
                argnums=(0, 1),
                expected_value=11.0,
                expected_grad=(gs.array([3.0, 4.0]), gs.array([1.0, 2.0])),
            ),
            dict(
                func=_cube_jacobian_sum,
                x=gs.array([1.0, 2.0]),
                expected_value=15.0,
                expected_grad=gs.array([6.0, 12.0]),
            ),
            dict(
                func=gs.linalg.matrix_norm,
                x=gs.array([[3.0, 0.0], [0.0, 4.0]]),
                point_ndims=2,
                expected_value=5.0,
                expected_grad=gs.array([[0.6, 0.0], [0.0, 0.8]]),
            ),
            dict(
                func=gs.linalg.vector_norm,
                x=gs.array([3.0, 4.0]),
                expected_value=5.0,
                expected_grad=gs.array([0.6, 0.8]),
            ),
            dict(
                func=gs.trace,
                x=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                point_ndims=2,
                expected_value=5.0,
                expected_grad=gs.eye(2),
            ),
            # the clipped entries get no gradient
            dict(
                func=_clip_sum,
                x=gs.array([0.0, 1.0]),
                expected_value=1.5,
                expected_grad=gs.array([0.0, 1.0]),
            ),
            # with S the square root, the gradient Y solves S Y + Y S = 1
            dict(
                func=_sqrtm_sum,
                x=gs.array([[4.0, 0.0], [0.0, 9.0]]),
                point_ndims=2,
                expected_value=5.0,
                expected_grad=gs.array([[1.0 / 4.0, 1.0 / 5.0], [1.0 / 5.0, 1.0 / 6.0]]),
            ),
            # the square root of diag(-4, -9) is diag(2i, 3i), and the gradient
            # is imaginary, with a sign depending on the complex convention
            dict(
                func=_real_sqrtm_sum,
                x=gs.array([[-4.0, 0.0], [0.0, -9.0]], dtype=gs.get_default_cdtype()),
                point_ndims=2,
                expected_value=0.0,
                expected_grad=(1j if pytorch_backend() else -1j)
                * gs.array([[1.0 / 4.0, 1.0 / 5.0], [1.0 / 5.0, 1.0 / 6.0]]),
            ),
            # the gradient gathers the weights at the indices
            dict(
                func=_weighted_scatter_add_sum,
                x=gs.array([1.0, 2.0, 3.0]),
                expected_value=10.0,
                expected_grad=gs.array([1.0, 3.0, 1.0]),
            ),
            dict(
                func=_diagonal_sum,
                x=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                point_ndims=2,
                expected_value=5.0,
                expected_grad=gs.eye(2),
            ),
            dict(
                func=_weighted_transpose_sum,
                x=gs.ones((2, 2)),
                point_ndims=2,
                expected_value=10.0,
                expected_grad=gs.array([[1.0, 3.0], [2.0, 4.0]]),
            ),
            dict(
                func=_sylvester_sum,
                x=gs.eye(2),
                point_ndims=2,
                expected_value=2.0,
                expected_grad=-gs.ones((2, 2)),
            ),
            dict(
                func=_sylvester_sum,
                x=gs.array([[1.0, 0.0], [0.0, 3.0]]),
                point_ndims=2,
                expected_value=7.0 / 6.0,
                expected_grad=-gs.array(
                    [[5.0 / 8.0, 1.0 / 3.0], [1.0 / 3.0, 13.0 / 72.0]]
                ),
            ),
        ]

        return data

    def value_and_grad_vec_test_data(self):
        data = [
            dict(
                func=_first_squared_plus_second,
                x=rand(2),
            ),
            dict(
                func=_squared_dist,
                x=rand(2),
                y=rand(2),
                argnums=1,
            ),
            dict(
                func=_squared_dist,
                x=rand(2),
                y=rand(2),
                argnums=(0, 1),
            ),
            dict(
                func=_dot_with_square,
                x=rand(2),
                y=rand(2),
                argnums=(0, 1),
            ),
            dict(
                func=_mat_squared_dist,
                x=rand(2, 2),
                y=rand(2, 2),
                argnums=(0, 1),
                point_ndims=2,
            ),
            dict(
                func=_mat_squared_dist,
                x=rand(2, 2),
                y=rand(2, 2),
                argnums=0,
                point_ndims=2,
            ),
            dict(
                func=_matvec_sum,
                x=rand(2, 2),
                y=rand(2),
                point_ndims=(2, 1),
                argnums=(0, 1),
            ),
            dict(
                func=_matvec_sum,
                x=rand(2, 2),
                y=rand(2),
                point_ndims=(2, 1),
                argnums=0,
            ),
            dict(
                func=_scalar_squared_dist,
                x=gs.array(0.7),
                y=gs.array(0.2),
                argnums=(0, 1),
                point_ndims=0,
            ),
            dict(
                func=_squared_dist_to_scalar,
                x=rand(2),
                y=gs.array(0.2),
                argnums=(0, 1),
                point_ndims=(1, 0),
            ),
            dict(func=gs.vecdot, x=rand(3), y=rand(3), argnums=(0, 1)),
            dict(func=_cube_jacobian_sum, x=rand(3)),
            dict(func=gs.linalg.matrix_norm, x=rand(2, 2), point_ndims=2),
            dict(func=functools.partial(gs.linalg.vector_norm, axis=-1), x=rand(3)),
            dict(func=gs.trace, x=rand(2, 2), point_ndims=2),
            dict(func=_weighted_scatter_add_sum, x=rand(3)),
            dict(func=_clip_sum, x=rand(3)),
            dict(func=_diagonal_sum, x=rand(2, 2), point_ndims=2),
            dict(func=_weighted_transpose_sum, x=rand(2, 2), point_ndims=2),
            dict(
                func=_expm_sum,
                x=rand(2, 2),
                point_ndims=2,
            ),
            dict(
                func=_logm_sum,
                x=rand(2, 2) + 2 * gs.eye(2),
                point_ndims=2,
            ),
            dict(
                func=_sqrtm_sum,
                x=rand(2, 2) + 2 * gs.eye(2),
                point_ndims=2,
            ),
        ]

        return self._vectorize(data, expected_name=("expected_value", "expected_grad"))

    @pytest.mark.smoke
    def jacobian_test_data(self):
        point = gs.array([gs.pi / 3, gs.pi])
        data = [
            dict(
                func=_sphere_immersion,
                x=point,
                expected=_sphere_immersion_jacobian(point),
            ),
            dict(
                func=_autodiff_first_component_jacobian,
                x=point,
                expected=_first_component_hessian(point),
            ),
            dict(
                func=_autodiff_first_component_hessian,
                x=point,
                expected=_first_component_third_derivative(point),
            ),
            dict(
                func=_autodiff_coordinates_product_hessian,
                x=gs.array([0.3, -1.2]),
                expected=gs.zeros((2, 2, 2)),
            ),
        ]
        return data

    def jacobian_vec_test_data(self):
        data = [
            dict(func=_sphere_immersion, x=rand(2)),
            dict(func=_autodiff_first_component_jacobian, x=rand(2)),
            dict(func=_autodiff_first_component_hessian, x=rand(2)),
            dict(func=_cube, x=gs.array(0.7), point_ndim=0),
            dict(func=_cube, x=rand(2, 2), point_ndim=2),
        ]
        return self._vectorize(
            data, get_expected_func=_get_derivative_func_getter("jacobian")
        )

    @pytest.mark.smoke
    def hessian_test_data(self):
        point = gs.array([gs.pi / 3, gs.pi])
        data = [
            dict(
                func=_first_component,
                x=point,
                expected=_first_component_hessian(point),
            ),
            dict(
                func=_square_and_product,
                x=gs.array([0.3, -1.2]),
                func_out_ndim=2,
                expected=gs.array([[[2.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]]),
            ),
            dict(
                func=_expm_sum,
                x=gs.zeros((2, 2)),
                point_ndim=2,
                expected=_half_squared_sum_hessian(2),
            ),
            dict(
                func=_logm_sum,
                x=gs.eye(2),
                point_ndim=2,
                expected=-_half_squared_sum_hessian(2),
            ),
        ]
        return data

    def hessian_vec_test_data(self):
        data = [
            dict(func=_first_component, x=rand(2)),
            dict(func=_square_and_product, x=rand(2), func_out_ndim=2),
            dict(func=_cube, x=gs.array(0.7), point_ndim=0),
            dict(func=_mat_sum_of_cubes, x=rand(2, 2), point_ndim=2),
        ]
        return self._vectorize(
            data, get_expected_func=_get_derivative_func_getter("hessian")
        )

    def value_and_jacobian_is_consistent_test_data(self):
        data = [
            dict(func=_sphere_immersion, x=rand(3, 2)),
            dict(func=_square_and_product, x=rand(2, 3, 2)),
            dict(func=_cube, x=rand(2, 3), point_ndim=0),
            dict(func=_cube, x=rand(2, 3, 2, 2), point_ndim=2),
        ]
        return data

    def jacobian_and_hessian_is_consistent_test_data(self):
        data = [
            dict(func=_first_component, x=rand(3, 2)),
            dict(func=_square_and_product, x=rand(2, 3, 2), func_out_ndim=2),
        ]
        return data

    def value_jacobian_and_hessian_is_consistent_test_data(self):
        data = [
            dict(func=_first_component, x=rand(3, 2)),
            dict(func=_square_and_product, x=rand(2, 3, 2), func_out_ndim=2),
        ]
        return data


class CustomGradientTestData(_ValueAndGradTestData):
    @pytest.mark.smoke
    def value_and_grad_test_data(self):
        """Test value_and_grad with custom gradient.

        NB: fake gradients are used to ensure the code goes through
        the expected path.
        """
        data = [
            dict(
                func=_sum_of_squares_with_fake_grad,
                x=gs.array([1.0, 3.0]),
                expected_value=10.0,
                expected_grad=gs.array([6.0, 18.0]),
            ),
            dict(
                func=_squared_dist_with_fake_grads,
                x=gs.array([1.0, 2.0]),
                y=gs.array([2.0, 3.0]),
                argnums=(0, 1),
                expected_value=2.0,
                expected_grad=(
                    gs.array([-6.0, -6.0]),
                    gs.array([6.0, 6.0]),
                ),
            ),
            dict(
                func=_mat_squared_dist_with_fake_grads,
                x=gs.array([[1.0, 3.0], [2.0, 3.0]]),
                y=gs.array([[2.0, 5.0], [0.0, 4.0]]),
                argnums=(0, 1),
                point_ndims=2,
                expected_value=10.0,
                expected_grad=(
                    gs.array([[-6.0, -12.0], [12.0, -6]]),
                    gs.array([[6.0, 12.0], [-12.0, 6.0]]),
                ),
            ),
            dict(
                func=_cubed_squared_sum_with_fake_grad,
                x=gs.array([1.0, 2.0]),
                expected_value=729.0,
                expected_grad=gs.array([1458.0, 2916.0]),
            ),
        ]

        return data

    def value_and_grad_vec_test_data(self):
        data = [
            dict(
                func=_sum_of_squares_with_fake_grad,
                x=rand(2),
            ),
            dict(
                func=_squared_dist_with_fake_grads,
                x=rand(2),
                y=rand(2),
                argnums=(0, 1),
            ),
            dict(
                func=_squared_dist_with_fake_grads,
                x=rand(2),
                y=rand(2),
                argnums=0,
            ),
            dict(
                func=_mat_squared_dist_with_fake_grads,
                x=rand(2, 2),
                y=rand(2, 2),
                argnums=0,
                point_ndims=2,
            ),
            dict(
                func=_mat_squared_dist_with_fake_grads,
                x=rand(2, 2),
                y=rand(2, 2),
                argnums=(0, 1),
                point_ndims=2,
            ),
            dict(
                func=_cubed_squared_sum_with_fake_grad,
                x=rand(2),
            ),
            dict(
                func=_mat_cubed_squared_sum_with_fake_grad,
                x=rand(2, 2),
                point_ndims=2,
            ),
        ]

        return self._vectorize(data, expected_name=("expected_value", "expected_grad"))
