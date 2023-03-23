import abc

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.pullback_metric import PullbackMetric
from tests.data_generation import TestData


class ImmersedSet(Manifold, abc.ABC):
    """Class for manifolds embedded in a vector space by an immersion.

    The manifold is represented with intrinsic coordinates, such that
    the immersion gives a parameterization of the manifold in these
    coordinates.

    Parameters
    ----------
    dim : int
        Dimension of the embedded manifold.
    """

    def __init__(self, dim, equip=True):
        super().__init__(
            dim=dim, shape=(dim,), default_coords_type="intrinsic", equip=equip
        )
        self.embedding_space = self._define_embedding_space()

    def _default_metric(self):
        return PullbackMetric

    @abc.abstractmethod
    def _define_embedding_space(self):
        """Define embedding space of the manifold.

        Returns
        -------
        embedding_space : Manifold
            Instance of Manifold.
        """

    @abc.abstractmethod
    def immersion(self, point):
        """Evaluate the immersion function at a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the immersed manifold.

        Returns
        -------
        immersion : array-like, shape=[..., dim_embedding]
            Immersion of the point.
        """

    def tangent_immersion(self, tangent_vec, base_point):
        """Evaluate the tangent immersion at a tangent vec and point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
        base_point : array-like, shape=[..., dim]
            Point in the immersed manifold.

        Returns
        -------
        tangent_vec_emb : array-like, shape=[..., dim_embedding]
        """
        jacobian_immersion = self.jacobian_immersion(base_point)
        return gs.matvec(jacobian_immersion, tangent_vec)

    def jacobian_immersion(self, base_point):
        """Evaluate the Jacobian of the immersion at a point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point in the immersed manifold.

        Returns
        -------
        jacobian_immersion : array-like, shape=[..., dim_embedding, dim]
        """
        return gs.autodiff.jacobian_vec(self.immersion)(base_point)

    def hessian_immersion(self, base_point):
        """Compute the Hessian of the immersion.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        hessian_immersion : array-like, shape=[..., embedding_dim, dim, dim]
            Hessian at the base point
        """
        return gs.autodiff.hessian_vec(
            self.immersion, func_out_ndim=self.embedding_space.dim
        )(base_point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError("`is_tangent` is not implemented yet")

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        raise NotImplementedError("`is_tangent` is not implemented yet")

    def projection(self, point):
        """Project a point to the embedded manifold.

        This is simply point, since we are in intrinsic coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in the embedding manifold.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point in the embedded manifold.
        """
        raise NotImplementedError("`projection` is not implemented yet")

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        This is simply the vector since we are in intrinsic coordinates.

        Parameters
        ----------
        vector : array-like, shape=[..., dim_embedding]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point in the embedded manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        raise NotImplementedError("`to_tangent` is not implemented yet")

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold according to some distribution.

        If the manifold is compact, preferably a uniform distribution will be used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., *point_shape]
            Points sampled on the manifold.
        """
        raise NotImplementedError("`random_point` is not implemented yet")


class CircleIntrinsic(ImmersedSet):
    def __init__(self, equip=True):
        super().__init__(dim=1, equip=equip)

    def immersion(self, point):
        return gs.hstack([gs.cos(point), gs.sin(point)])

    def _define_embedding_space(self):
        return Euclidean(dim=self.dim + 1)


class SphereIntrinsic(ImmersedSet):
    def __init__(self, equip=True):
        super().__init__(dim=2, equip=equip)

    def immersion(self, point):
        theta = point[..., 0]
        phi = point[..., 1]
        return gs.stack(
            [
                gs.cos(phi) * gs.sin(theta),
                gs.sin(phi) * gs.sin(theta),
                gs.cos(theta),
            ],
            axis=-1,
        )

    def _define_embedding_space(self):
        return Euclidean(dim=self.dim + 1)


def _expected_jacobian_circle_immersion(point):
    return gs.stack(
        [
            -gs.sin(point),
            gs.cos(point),
        ],
        axis=-2,
    )


def _expected_jacobian_sphere_immersion(point):
    theta = point[..., 0]
    phi = point[..., 1]
    jacobian = gs.array(
        [
            [gs.cos(phi) * gs.cos(theta), -gs.sin(phi) * gs.sin(theta)],
            [gs.sin(phi) * gs.cos(theta), gs.cos(phi) * gs.sin(theta)],
            [-gs.sin(theta), 0.0],
        ]
    )
    return jacobian


def _expected_hessian_sphere_immersion(point):
    theta = point[..., 0]
    phi = point[..., 1]
    hessian_immersion_x = gs.array(
        [
            [-gs.sin(theta) * gs.cos(phi), -gs.cos(theta) * gs.sin(phi)],
            [-gs.cos(theta) * gs.sin(phi), -gs.sin(theta) * gs.cos(phi)],
        ]
    )
    hessian_immersion_y = gs.array(
        [
            [-gs.sin(theta) * gs.sin(phi), gs.cos(theta) * gs.cos(phi)],
            [gs.cos(theta) * gs.cos(phi), -gs.sin(theta) * gs.sin(phi)],
        ]
    )
    hessian_immersion_z = gs.array([[-gs.cos(theta), 0.0], [0.0, 0.0]])
    hessian_immersion = gs.stack(
        [hessian_immersion_x, hessian_immersion_y, hessian_immersion_z], axis=0
    )
    return hessian_immersion


def _expected_circle_metric_matrix(point):
    mat = gs.array([[1.0]])
    return mat


def _expected_sphere_metric_matrix(point):
    theta = point[..., 0]
    mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])
    return mat


def _expected_inverse_circle_metric_matrix(point):
    mat = gs.array([[1.0]])
    return mat


def _expected_inverse_sphere_metric_matrix(point):
    theta = point[..., 0]
    mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** (-2)]])
    return mat


class PullbackMetricTestData(TestData):

    Metric = PullbackMetric

    def sphere_immersion_test_data(self):
        smoke_data = [
            dict(
                space=SphereIntrinsic(equip=False),
                point=gs.array([0.0, 0.0]),
                expected=gs.array([0.0, 0.0, 1.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                point=gs.array([gs.pi, 0.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                point=gs.array([gs.pi / 2.0, gs.pi]),
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sphere_immersion_and_spherical_to_extrinsic_test_data(self):
        smoke_data = [
            dict(space=SphereIntrinsic(equip=False), point=gs.array([0.0, 0.0]))
        ]
        return self.generate_tests(smoke_data)

    def tangent_immersion_test_data(self):
        smoke_data = [
            dict(
                space=CircleIntrinsic(equip=False),
                tangent_vec=gs.array([1.0]),
                point=gs.array([0.0]),
                expected=gs.array([0.0, 1.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec=gs.array([1.0, 0.0]),
                point=gs.array([gs.pi / 2.0, gs.pi / 2.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec=gs.array([0.0, 1.0]),
                point=gs.array([gs.pi / 2.0, gs.pi / 2.0]),
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec=gs.array([1.0, 0.0]),
                point=gs.array([gs.pi / 2.0, 0.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec=gs.array([0.0, 1.0]),
                point=gs.array([gs.pi / 2.0, 0.0]),
                expected=gs.array([0.0, 1.0, 0.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def jacobian_immersion_test_data(self):
        smoke_data = [
            dict(
                space=CircleIntrinsic(equip=False),
                pole=gs.array([0.0]),
                expected_func=_expected_jacobian_circle_immersion,
            ),
            dict(
                space=CircleIntrinsic(equip=False),
                pole=gs.array([0.2]),
                expected_func=_expected_jacobian_circle_immersion,
            ),
            dict(
                space=CircleIntrinsic(equip=False),
                pole=gs.array([4.0]),
                expected_func=_expected_jacobian_circle_immersion,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                pole=gs.array([0.0, 0.0]),
                expected_func=_expected_jacobian_sphere_immersion,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                pole=gs.array([0.22, 0.1]),
                expected_func=_expected_jacobian_sphere_immersion,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                pole=gs.array([0.1, 0.88]),
                expected_func=_expected_jacobian_sphere_immersion,
            ),
        ]
        return self.generate_tests(smoke_data)

    def hessian_sphere_immersion_test_data(self):
        smoke_data = [
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([0.0, 0.0]),
                expected_func=_expected_hessian_sphere_immersion,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([0.22, 0.1]),
                expected_func=_expected_hessian_sphere_immersion,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([0.1, 0.88]),
                expected_func=_expected_hessian_sphere_immersion,
            ),
        ]
        return self.generate_tests(smoke_data)

    def second_fundamental_form_sphere_test_data(self):
        smoke_data = [
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.22, 0.1])),
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.1, 0.88])),
        ]
        return self.generate_tests(smoke_data)

    def second_fundamental_form_circle_test_data(self):
        smoke_data = [
            dict(space=CircleIntrinsic(equip=False), base_point=gs.array([0.22])),
            dict(space=CircleIntrinsic(equip=False), base_point=gs.array([0.88])),
        ]
        return self.generate_tests(smoke_data)

    def mean_curvature_vector_norm_sphere_test_data(self):
        smoke_data = [
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.22, 0.1])),
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.1, 0.88])),
        ]
        return self.generate_tests(smoke_data)

    def mean_curvature_vector_norm_circle_test_data(self):
        smoke_data = [
            dict(space=CircleIntrinsic(equip=False), base_point=gs.array([0.1])),
            dict(space=CircleIntrinsic(equip=False), base_point=gs.array([0.88])),
        ]
        return self.generate_tests(smoke_data)

    def parallel_transport_and_hypersphere_parallel_transport_test_data(self):
        smoke_data = [
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec_a=gs.array([0.0, 1.0]),
                tangent_vec_b=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
            )
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                space=CircleIntrinsic(equip=False),
                base_point=gs.array([0.0]),
                expected_func=_expected_circle_metric_matrix,
            ),
            dict(
                space=CircleIntrinsic(equip=False),
                base_point=gs.array([1.0]),
                expected_func=_expected_circle_metric_matrix,
            ),
            dict(
                space=CircleIntrinsic(equip=False),
                base_point=gs.array([4.0]),
                expected_func=_expected_circle_metric_matrix,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([0.0, 0.0]),
                expected_func=_expected_sphere_metric_matrix,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([1.0, 1.0]),
                expected_func=_expected_sphere_metric_matrix,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([0.3, 0.8]),
                expected_func=_expected_sphere_metric_matrix,
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_and_hypersphere_inner_product_test_data(self):
        smoke_data = [
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec_a=gs.array([0.0, 1.0]),
                tangent_vec_b=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec_a=gs.array([0.4, 1.0]),
                tangent_vec_b=gs.array([0.2, 0.6]),
                base_point=gs.array([gs.pi / 2.0, 0.1]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inverse_metric_matrix_test_data(self):
        smoke_data = [
            dict(
                space=CircleIntrinsic(equip=False),
                base_point=gs.array([0.6]),
                expected_func=_expected_inverse_circle_metric_matrix,
            ),
            dict(
                space=CircleIntrinsic(equip=False),
                base_point=gs.array([0.8]),
                expected_func=_expected_inverse_circle_metric_matrix,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([0.6, -1.0]),
                expected_func=_expected_inverse_sphere_metric_matrix,
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                base_point=gs.array([0.8, -0.8]),
                expected_func=_expected_inverse_sphere_metric_matrix,
            ),
        ]
        return self.generate_tests(smoke_data)

    def sphere_inner_product_derivative_matrix_test_data(self):
        smoke_data = [
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.6, -1.0])),
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.8, -0.8])),
        ]
        return self.generate_tests(smoke_data)

    def christoffels_and_hypersphere_christoffels_test_data(self):
        smoke_data = [
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.1, 0.2])),
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.7, 0.233])),
        ]
        return self.generate_tests(smoke_data)

    def christoffels_sphere_test_data(self):
        smoke_data = [
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.1, 0.2])),
            dict(space=SphereIntrinsic(equip=False), base_point=gs.array([0.7, 0.233])),
        ]
        return self.generate_tests(smoke_data)

    def christoffels_circle_test_data(self):
        smoke_data = [
            dict(space=CircleIntrinsic(equip=False), base_point=gs.array([0.1])),
            dict(space=CircleIntrinsic(equip=False), base_point=gs.array([0.7])),
        ]
        return self.generate_tests(smoke_data)

    def exp_and_hypersphere_exp_test_data(self):
        smoke_data = [
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
            ),
            dict(
                space=SphereIntrinsic(equip=False),
                tangent_vec=gs.array([0.4, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.1]),
            ),
        ]
        return self.generate_tests(smoke_data)
